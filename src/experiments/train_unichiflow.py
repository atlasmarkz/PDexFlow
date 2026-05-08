import os
import time

import hydra
import torch
from omegaconf import DictConfig
from torch import amp
from torch.nn.utils import clip_grad_norm_
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


config_path = os.path.abspath("configs")
config_name = "unichiflow_backbone.yaml"


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
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

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

    def _check_and_clip_gradients(parameters):
        """Fail-loud NaN/Inf detection. Returns True if gradients are finite and safe to step.

        Previous behavior silently replaced NaN/Inf with 0, which hid genuine numerical
        failures (the next forward would diverge again on the same inputs). Now we detect
        any non-finite grad, log it, and tell the caller to skip the opt step — the
        surrounding loop already has bad_step counter + LR decay plumbing.
        """
        has_bad = False
        for p in parameters:
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                has_bad = True
                break
        if has_bad:
            return False
        if max_grad_value > 0:
            for p in parameters:
                if p.grad is not None:
                    p.grad.clamp_(min=-max_grad_value, max=max_grad_value)
        return True

    # Best checkpoint keeper (old name: EarlyStopping — the stop branch almost never fires in practice).
    es_cfg = cfg.train.get('early_stopping', {})
    early_stopping = BestCheckpointKeeper(
        patience=int(es_cfg.get('patience', 10000)),
        min_delta=float(es_cfg.get('min_delta', 0.0)),
        path=os.path.join(ckpt_dir, "best_model.pt"),
        verbose=False
    )
    smooth_loss = None
    smooth_alpha = 0.99

    model.train()
    pbar = tqdm(range(1, int(cfg.train.max_iters) + 1))
    optim.zero_grad(set_to_none=True)
    global_step = 0
    bad_step = 0

    def _decay_lr(factor: float = 0.5, min_lr: float = 1e-7):
        for g in optim.param_groups:
            old_lr = g['lr']
            new_lr = max(old_lr * factor, min_lr)
            g['lr'] = new_lr
        return optim.param_groups[0]['lr']

    target_sym_weight = float(cfg.flow.get("sym_weight", 1.0))
    # Configurable warmup
    warmup_enable = bool(cfg.flow.get("sym_warmup_enable", True))
    warmup_start = int(cfg.flow.get("sym_warmup_start", 500))
    warmup_end = int(cfg.flow.get("sym_warmup_end", 2000))

    def get_sym_weight(s):
        if not warmup_enable: return target_sym_weight
        if s < warmup_start: return 0.0
        if s >= warmup_end: return target_sym_weight
        return target_sym_weight * (s - warmup_start) / (warmup_end - warmup_start)

    for step in pbar:
        batch = du.recursive_to(next(it), device)
        # NOTE: Coordinate scaling (×0.1) is now performed inside UniChiFlowModel
        # (corrupt_batch / sample) via self.coord_scale. Scripts must pass raw PDB
        # angstroms in batch['pos_heavyatom']; do NOT pre-scale here.

        current_sym_weight = get_sym_weight(step)

        with amp.autocast(device_type=amp_device, enabled=use_amp):
            loss_dict = model(batch, sym_weight_override=current_sym_weight)
            loss = loss_dict['loss'] / float(grad_accum_steps)

        # Skip step if any loss component is non-finite
        _finite_losses = all(torch.isfinite(v) for v in loss_dict.values()) and torch.isfinite(loss)
        if not _finite_losses:
            pbar.set_description("non-finite loss_dict, skipping step")
            optim.zero_grad(set_to_none=True)
            bad_step += 1
            if bad_step % 10 == 0:
                lr_now = _decay_lr()
                pbar.set_description(f"decay lr -> {lr_now:.2e} (loss nan)")
            continue

        if not torch.isfinite(loss):
            pbar.set_description("non-finite loss, skipping step")
            optim.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if step % grad_accum_steps != 0:
            continue

        grad = None
        if use_amp:
            scaler.unscale_(optim)
            if not _check_and_clip_gradients(list(model.parameters())):
                pbar.set_description("non-finite grad (pre-clip), skipping opt step")
                optim.zero_grad(set_to_none=True)
                scaler.update()
                bad_step += 1
                if bad_step % 10 == 0:
                    lr_now = _decay_lr()
                    pbar.set_description(f"decay lr -> {lr_now:.2e} (grad nan)")
                continue
            grad = clip_grad_norm_(model.parameters(), float(cfg.train.max_grad_norm))
            if not torch.isfinite(grad):
                pbar.set_description("non-finite grad, skipping opt step")
                # drop this accumulation; reset scaler to avoid stale state
                optim.zero_grad(set_to_none=True)
                scaler.update()
                bad_step += 1
                if bad_step % 10 == 0:
                    lr_now = _decay_lr()
                    pbar.set_description(f"decay lr -> {lr_now:.2e} (grad nan)")
                continue
            scaler.step(optim)
            scaler.update()
        else:
            if not _check_and_clip_gradients(list(model.parameters())):
                pbar.set_description("non-finite grad (pre-clip), skipping opt step")
                optim.zero_grad(set_to_none=True)
                bad_step += 1
                if bad_step % 10 == 0:
                    lr_now = _decay_lr()
                    pbar.set_description(f"decay lr -> {lr_now:.2e} (grad nan)")
                continue
            grad = clip_grad_norm_(model.parameters(), float(cfg.train.max_grad_norm))
            if not torch.isfinite(grad):
                pbar.set_description("non-finite grad, skipping opt step")
                optim.zero_grad(set_to_none=True)
                bad_step += 1
                if bad_step % 10 == 0:
                    lr_now = _decay_lr()
                    pbar.set_description(f"decay lr -> {lr_now:.2e} (grad nan)")
                continue
            optim.step()
        optim.zero_grad(set_to_none=True)

        global_step += 1

        # Update smoothed loss for early stopping
        raw_loss = loss_dict['loss'].item()
        if smooth_loss is None:
            smooth_loss = raw_loss
        else:
            smooth_loss = smooth_alpha * smooth_loss + (1 - smooth_alpha) * raw_loss

        pbar.set_description(
            f"loss {raw_loss:.4f} (sm: {smooth_loss:.4f}) | "
            f"cfm {loss_dict['cfm_loss'].item():.3f}*{loss_dict.get('w_cfm', torch.tensor(0.0)).item():.1e} | "
            f"sym {loss_dict['sym_loss'].item():.3f}*{loss_dict.get('w_sym', torch.tensor(0.0)).item():.1e} | "
            f"seq {loss_dict['seq_loss'].item():.3f}*{loss_dict.get('w_seq', torch.tensor(0.0)).item():.1e} | "
            f"g {grad:.1f}"
        )

        writer.add_scalar('train/loss', raw_loss, global_step)
        writer.add_scalar('train/smooth_loss', smooth_loss, global_step)
        writer.add_scalar('train/cfm_loss', loss_dict['cfm_loss'].item(), global_step)
        writer.add_scalar('train/sym_loss', loss_dict['sym_loss'].item(), global_step)
        writer.add_scalar('train/seq_loss', loss_dict['seq_loss'].item(), global_step)
        writer.add_scalar('train/grad_norm', grad, global_step)
        writer.add_scalar('train/lr', optim.param_groups[0]['lr'], global_step)

        bad_step = 0  # reset on successful step

        # Track best checkpoint; honor the stop flag if set (rare in practice).
        early_stopping(smooth_loss, model, cfg, optim, global_step)
        if early_stopping.early_stop:
            print(f"\nBest-loss plateau at step {step}. Best smoothed loss: {early_stopping.best_loss:.6f}")
            break

        if step % int(cfg.train.save_freq) == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{step}.pt")
            torch.save({
                'config': cfg,
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'step': step,
            }, ckpt_path)


if __name__ == "__main__":
    main()
