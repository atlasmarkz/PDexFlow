import math
import os
import time

# Set ESM cache directory relative to the project root (../../cache/esm)
# We use abspath(__file__) to ensure it works regardless of the current working directory (e.g. when using Hydra).
_esm_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'cache', 'esm'))
os.makedirs(_esm_cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = _esm_cache_dir

import hydra
import torch
from omegaconf import DictConfig
from torch import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import utils as du
from src.experiments.components import (
    build_dataloader,
    build_dataset,
    build_model,
    build_optimizer,
    inf_iterator,
)
from src.experiments.train_step import StepConfig, do_train_step


_RAD2DEG = 180.0 / math.pi


config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
config_name = "pdexflow_training.yaml"


class BestCheckpointKeeper:
    """Tracks the best-so-far smoothed loss and writes a single `best_model.pt`.

    Originally named `EarlyStopping`, but in practice `patience` is set so high
    (>= max_iters) that the stop branch never fires — the class is really a
    "keep-the-best-checkpoint-seen" helper. Renamed so callers don't assume
    training can actually terminate from here. If the patience counter is ever
    exceeded the flag is still set, but the training loop treats that as a hint,
    not a hard stop.
    """
    def __init__(self, patience=1000, min_delta=0.0, path='checkpoint.pt', verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path
        self.verbose = verbose

    def __call__(self, val_loss, model, cfg, optimizer, step):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, cfg, optimizer, step)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose and self.counter % 100 == 0:
                print(f'BestCheckpointKeeper: no improvement for {self.counter}/{self.patience} steps')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, cfg, optimizer, step)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, cfg, optimizer, step):
        if self.verbose:
            print(f'Loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
            'config': cfg,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'loss': val_loss
        }, self.path)


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    du.seed_all(int(cfg.train.seed))

    device = str(cfg.train.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    run_name = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    out_dir = os.path.join(str(cfg.train.logdir), run_name)
    writer = SummaryWriter(log_dir=out_dir)
    model_dir = os.path.join("outputs", "checkpoints", run_name, "model")
    os.makedirs(model_dir, exist_ok=True)

    ds = build_dataset(cfg, kind="pepmerge")
    loader = build_dataloader(ds, cfg, eight=True)
    it = inf_iterator(loader)

    model = build_model(cfg, include_multitask_weights=True).to(device)

    optim = build_optimizer(model, cfg, trainable_only=False)
    use_amp = device.startswith("cuda") and bool(cfg.train.get("use_amp", True))
    # autocast device_type and GradScaler device must match the actual training
    # device so CPU/MPS smoke runs don't fail at construction time.
    amp_device = "cuda" if device.startswith("cuda") else "cpu"
    scaler = amp.GradScaler(enabled=use_amp, device=amp_device)

    grad_accum_steps = int(cfg.train.get("grad_accum_steps", 1))
    grad_accum_steps = max(1, grad_accum_steps)

    max_grad_value = float(cfg.train.get("max_grad_value", 0.0))

    step_cfg = StepConfig(
        use_amp=use_amp,
        amp_device=amp_device,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=float(cfg.train.max_grad_norm),
        max_grad_value=max_grad_value,
    )

    # Best checkpoint keeper (old name: EarlyStopping — the stop branch almost never fires in practice).
    es_cfg = cfg.train.get('early_stopping', {})
    early_stopping = BestCheckpointKeeper(
        patience=int(es_cfg.get('patience', 10000)),
        min_delta=float(es_cfg.get('min_delta', 0.0)),
        path=os.path.join(model_dir, "best_model.pt"),
        verbose=False
    )
    smooth_loss = None
    smooth_alpha = 0.99

    model.train()
    pbar = tqdm(range(1, int(cfg.train.max_iters) + 1))
    optim.zero_grad(set_to_none=True)
    global_step = 0
    bad_step = 0
    total_skips = 0  # cumulative skipped opt steps (loss or grad non-finite), for TB skip_rate

    def _decay_lr(factor: float = 0.5, min_lr: float = 1e-7):
        for g in optim.param_groups:
            old_lr = g['lr']
            new_lr = max(old_lr * factor, min_lr)
            g['lr'] = new_lr
        return optim.param_groups[0]['lr']

    target_sym_weight = float(cfg.flow.get("sym_weight", 1.0))
    sym_warmup_enable = bool(cfg.flow.get("sym_warmup_enable", False))
    sym_warmup_start = float(cfg.flow.get("sym_warmup_start_weight", target_sym_weight))
    sym_warmup_steps = max(1, int(cfg.flow.get("sym_warmup_steps", 1)))

    def _scheduled_sym_weight(step_i: int) -> float:
        if not sym_warmup_enable:
            return target_sym_weight
        alpha = min(1.0, max(0.0, float(step_i) / float(sym_warmup_steps)))
        return sym_warmup_start + (target_sym_weight - sym_warmup_start) * alpha

    for step in pbar:
        batch = du.recursive_to(next(it), device)
        chir = batch.get('chirality', None)
        if isinstance(chir, torch.Tensor):
            writer.add_scalar('train/ld_ratio', chir.float().mean().item(), step)
        current_sym_weight = _scheduled_sym_weight(step)
        writer.add_scalar('train/w_sym_override', current_sym_weight, step)

        # Debug: print batch data ranges every 5 steps
        if step % 5 == 0:
            pos = batch['pos_heavyatom']
            aa = batch['aa']
            gen_mask = batch['generate_mask']
            res_mask = batch['res_mask']
            print(f"[Step {step}] BATCH | pos_range=[{pos.min():.3f}, {pos.max():.3f}] pos_finite={pos.isfinite().all().item()} | aa_unique={aa.unique().numel()} | pep_len={gen_mask.sum().item()} | res_len={res_mask.sum().item()} | device={pos.device}")

        outcome = do_train_step(
            batch=batch,
            model=model,
            optim=optim,
            scaler=scaler,
            cfg=step_cfg,
            step=step,
            sym_weight_override=current_sym_weight,
        )

        if outcome.kind == "accum":
            continue  # mid grad-accum, no opt step this iteration

        if outcome.kind == "skip_loss_nonfinite":
            # Log which loss component is non-finite for debugging
            cfm = outcome.components.get('cfm_loss', float('nan'))
            sym = outcome.components.get('sym_loss', float('nan'))
            seq = outcome.components.get('seq_loss', float('nan'))
            total = outcome.loss
            print(f"[Step {step}] SKIP loss_nonfinite | cfm={cfm} sym={sym} seq={seq} | total={total}")
            pbar.set_description("non-finite loss_dict, skipping step")
            bad_step += 1
            total_skips += 1
            if bad_step % 10 == 0:
                lr_now = _decay_lr()
                pbar.set_description(f"decay lr -> {lr_now:.2e} (loss nan)")
            writer.add_scalar('train/skip_rate', total_skips / step, step)
            writer.add_scalar('train/bad_step_streak', bad_step, step)
            continue

        if outcome.kind == "skip_grad_nonfinite":
            cfm = outcome.components.get('cfm_loss', float('nan'))
            sym = outcome.components.get('sym_loss', float('nan'))
            seq = outcome.components.get('seq_loss', float('nan'))
            total = outcome.loss
            print(f"[Step {step}] SKIP grad_nonfinite | cfm={cfm} sym={sym} seq={seq} | total={total}")
            pbar.set_description("non-finite grad (pre-clip), skipping opt step")
            bad_step += 1
            total_skips += 1
            if bad_step % 10 == 0:
                lr_now = _decay_lr()
                pbar.set_description(f"decay lr -> {lr_now:.2e} (grad nan)")
            writer.add_scalar('train/skip_rate', total_skips / step, step)
            writer.add_scalar('train/bad_step_streak', bad_step, step)
            continue

        # outcome.kind == "ok"
        global_step += 1
        raw_loss = outcome.loss
        grad = outcome.grad_norm
        components = outcome.components

        if step % 5 == 0:
            print(f"[Step {step}] OK | loss={raw_loss:.4f} | cfm={components.get('cfm_loss', 0):.4f} sym={components.get('sym_loss', 0):.4f} seq={components.get('seq_loss', 0):.4f} | grad={grad:.4f}")

        # Update smoothed loss for early stopping
        if smooth_loss is None:
            smooth_loss = raw_loss
        else:
            smooth_loss = smooth_alpha * smooth_loss + (1 - smooth_alpha) * raw_loss

        pbar.set_description(
            f"loss {raw_loss:.4f} (sm: {smooth_loss:.4f}) | "
            f"cfm {components['cfm_loss']:.3f}*{components.get('w_cfm', 0.0):.1e} | "
            f"sym {components['sym_loss']:.3f}*{components.get('w_sym', 0.0):.1e} | "
            f"seq {components['seq_loss']:.3f}*{components.get('w_seq', 0.0):.1e} | "
            f"g {grad:.1f}"
        )

        writer.add_scalar('train/loss', raw_loss, global_step)
        writer.add_scalar('train/smooth_loss', smooth_loss, global_step)
        writer.add_scalar('train/cfm_loss', components['cfm_loss'], global_step)
        writer.add_scalar('train/sym_loss', components['sym_loss'], global_step)
        writer.add_scalar('train/seq_loss', components['seq_loss'], global_step)
        writer.add_scalar('train/grad_norm', grad, global_step)
        writer.add_scalar('train/lr', optim.param_groups[0]['lr'], global_step)
        # Interpretable diagnostic metrics (computed under no_grad in the model;
        # radians → degrees conversion only happens here, not in the hot forward path).
        if 'aa_accuracy' in components:
            writer.add_scalar('train/aa_accuracy', components['aa_accuracy'], global_step)
        if 'trans_mae_ang' in components:
            writer.add_scalar('train/trans_mae_ang', components['trans_mae_ang'], global_step)
        if 'rot_err_rad' in components:
            writer.add_scalar('train/rot_err_deg', components['rot_err_rad'] * _RAD2DEG, global_step)
        if 'torsion_err_rad' in components:
            writer.add_scalar('train/torsion_err_deg', components['torsion_err_rad'] * _RAD2DEG, global_step)
        # skip health — indexed by `step` (attempts) not `global_step` (successful steps)
        writer.add_scalar('train/skip_rate', total_skips / step, step)
        writer.add_scalar('train/bad_step_streak', bad_step, step)

        bad_step = 0  # reset on successful step

        # Track best checkpoint; honor the stop flag if set (rare in practice).
        early_stopping(smooth_loss, model, cfg, optim, global_step)
        if early_stopping.early_stop:
            print(f"\nBest-loss plateau at step {step}. Best smoothed loss: {early_stopping.best_loss:.6f}")
            break

        if step % int(cfg.train.save_freq) == 0:
            ckpt_path = os.path.join(model_dir, f"{step}.pt")
            torch.save({
                'config': cfg,
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'step': step,
            }, ckpt_path)


if __name__ == "__main__":
    main()
