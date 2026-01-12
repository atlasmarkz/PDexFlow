import os
import time

import hydra
import torch
from omegaconf import DictConfig
from torch import amp
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from unichiflow.data.pep_dataloader import PaddingCollate, TwinUniversePepDataset
from unichiflow.data import utils as du
from unichiflow.models.chiralformer import UniChiFlowModel


config_path = os.path.join(os.path.dirname(__file__), "..", "configs")
config_name = "unichiflow_pepmerge.yaml"


def _inf_iterator(loader):
    while True:
        for batch in loader:
            yield batch


class EarlyStopping:
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
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
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

    ds = TwinUniversePepDataset(
        structure_dir=str(cfg.data.structure_dir),
        cache_dir=str(cfg.data.cache_dir),
        split_name=f"pepmerge_{cfg.data.split}",
        mirror_prob=float(cfg.data.twin_universe_prob),
        reset=bool(cfg.data.reset_cache),
        deterministic_pairs=bool(cfg.data.get("deterministic_pairs", False)),
    )
    loader = DataLoader(ds, batch_size=int(cfg.train.batch_size), shuffle=True, num_workers=int(cfg.train.num_workers),
                        collate_fn=PaddingCollate(eight=True), pin_memory=True)
    it = _inf_iterator(loader)

    model = UniChiFlowModel(
        d_model=int(cfg.model.d_model),
        n_layers=int(cfg.model.n_layers),
        n_heads=int(cfg.model.n_heads),
        dropout=float(cfg.model.dropout),
        max_len=int(cfg.model.max_len),
        aatype_vocab=int(cfg.model.aatype_vocab),
        sigma_trans=float(cfg.flow.sigma_trans),
        chi_lora=cfg.get("chi_lora"),
        esm=cfg.get("esm"),
        cfm_weight=float(cfg.flow.get("cfm_weight", 1.0)),
        sym_weight=float(cfg.flow.get("sym_weight", 1.0)),
        seq_weight=float(cfg.flow.get("seq_weight", 1.0)),
        angle_dim=int(cfg.model.get("angle_dim", 5)),
        use_ot=bool(cfg.flow.get("use_ot", True)),
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.optimizer.lr), weight_decay=float(cfg.train.optimizer.weight_decay))
    use_amp = device.startswith("cuda") and bool(cfg.train.get("use_amp", True))
    scaler = amp.GradScaler(enabled=use_amp, device="cuda")

    grad_accum_steps = int(cfg.train.get("grad_accum_steps", 1))
    grad_accum_steps = max(1, grad_accum_steps)

    max_grad_value = float(cfg.train.get("max_grad_value", 0.0))

    def _sanitize_gradients(parameters):
        """Replace NaN/Inf grads and optionally clip by value before norm clipping."""
        for p in parameters:
            if p.grad is None:
                continue
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if max_grad_value > 0:
                p.grad.clamp_(min=-max_grad_value, max=max_grad_value)

    # Early Stopping setup
    es_cfg = cfg.train.get('early_stopping', {})
    early_stopping = EarlyStopping(
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
    warmup_steps = 2000

    def get_sym_weight(s):
        if s < 500: return 0.0
        if s >= warmup_steps: return target_sym_weight
        return target_sym_weight * (s - 500) / (warmup_steps - 500)

    for step in pbar:
        batch = du.recursive_to(next(it), device)

        # Normalize coordinates to improve numerical stability (Issue #5)
        if 'pos_heavyatom' in batch:
            batch['pos_heavyatom'] = batch['pos_heavyatom'] * 0.1

        current_sym_weight = get_sym_weight(step)

        with amp.autocast(device_type="cuda", enabled=use_amp):
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
            _sanitize_gradients(model.parameters())
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
            _sanitize_gradients(model.parameters())
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
            f"loss {raw_loss:.4f} (sm: {smooth_loss:.4f}) | cfm {loss_dict['cfm_loss'].item():.3f} | sym {loss_dict['sym_loss'].item():.3f} | seq {loss_dict['seq_loss'].item():.3f} | g {grad:.1f}"
        )

        writer.add_scalar('train/loss', raw_loss, global_step)
        writer.add_scalar('train/smooth_loss', smooth_loss, global_step)
        writer.add_scalar('train/cfm_loss', loss_dict['cfm_loss'].item(), global_step)
        writer.add_scalar('train/sym_loss', loss_dict['sym_loss'].item(), global_step)
        writer.add_scalar('train/seq_loss', loss_dict['seq_loss'].item(), global_step)
        writer.add_scalar('train/grad_norm', grad, global_step)
        writer.add_scalar('train/lr', optim.param_groups[0]['lr'], global_step)

        bad_step = 0  # reset on successful step

        # Check early stopping
        early_stopping(smooth_loss, model, cfg, optim, global_step)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at step {step}. Best smoothed loss: {early_stopping.best_loss:.6f}")
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
