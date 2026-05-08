import os
import sys
import time
import hydra
import torch
from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# Add parent directory to path if package is not installed
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data import utils as du
from src.experiments.components import (
    build_dataloader,
    build_dataset,
    build_model,
    build_optimizer,
    inf_iterator,
    load_initial_weights,
)

config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
config_name = "unichiflow_lora.yaml"

@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    du.seed_all(int(cfg.train.seed))

    device = str(cfg.train.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    run_name = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    out_dir = os.path.join(str(cfg.train.logdir), run_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ds = build_dataset(cfg, kind="real_d")
    loader = build_dataloader(ds, cfg, eight=True)
    it = inf_iterator(loader)

    model = build_model(cfg, include_multitask_weights=False).to(device)
    model = load_initial_weights(model, cfg, device)

    # If freeze_backbone is enabled, only train LoRA parameters
    if cfg.chi_lora.enable and cfg.chi_lora.freeze_backbone:
        print("Freezing backbone, training only LoRA parameters.")
        # UniChiFlowModel._apply_chi_lora performs the actual freeze in __init__;
        # this branch is kept for the log line and so the optimizer sees the
        # post-freeze requires_grad state when build_optimizer filters.

    # Compute trainable_params for downstream _check_and_clip_gradients/clip_grad_norm_;
    # build_optimizer filters requires_grad internally, so the two stay in sync.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")

    optim = build_optimizer(model, cfg, trainable_only=True)
    
    max_grad_value = float(cfg.train.get("max_grad_value", 5.0))

    def _check_and_clip_gradients(parameters):
        """Fail-loud NaN/Inf detection. Returns True iff all grads are finite.

        Replaced the old silent nan_to_num_ path: zeroing bad grads hides the fact
        that the forward just produced NaN, so the next batch reuses the same
        poisoned parameters and diverges again. Now callers skip the opt step
        and log instead.
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

    model.train()
    pbar = tqdm(range(1, int(cfg.train.max_iters) + 1))
    
    best_loss = float('inf')
    
    for step in pbar:
        try:
            batch = du.recursive_to(next(it), device)
            # Coordinate scaling is handled inside UniChiFlowModel.corrupt_batch via
            # self.coord_scale. Scripts pass raw angstroms; no manual pre-scale here.

            loss_dict = model(batch)
            loss = loss_dict['loss']

            # Skip step if any loss component is non-finite
            if not torch.isfinite(loss):
                print(f"Non-finite loss at step {step}: {loss_dict}")
                optim.zero_grad(set_to_none=True)
                continue
            
            optim.zero_grad(set_to_none=True)
            loss.backward()

            if not _check_and_clip_gradients(trainable_params):
                print(f"Non-finite gradient detected at step {step} (pre-clip), skipping optimizer step")
                optim.zero_grad(set_to_none=True)
                continue
            grad = clip_grad_norm_(trainable_params, float(cfg.train.max_grad_norm))

            if not torch.isfinite(grad):
                print(f"Non-finite gradient at step {step}: {grad}")
                optim.zero_grad(set_to_none=True)
                continue
                
            optim.step()

            # Update smoothed loss for logging
            loss_val = loss.item()
            
            # Safe access to loss components
            trans_loss = loss_dict.get('cfm_loss', torch.tensor(0.0)).item()
            sym_loss = loss_dict.get('sym_loss', torch.tensor(0.0)).item()
            seq_loss = loss_dict.get('seq_loss', torch.tensor(0.0)).item()

            pbar.set_description(
                f"loss {loss_val:.4f} | cfm {trans_loss:.3f} | sym {sym_loss:.3f} | seq {seq_loss:.3f} | g {grad:.1f}"
            )
            
            # Save best model
            if loss_val < best_loss:
                best_loss = loss_val
                ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
                torch.save({
                    'config': cfg,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'step': step,
                    'loss': best_loss
                }, ckpt_path)

            if step % int(cfg.train.save_freq) == 0:
                ckpt_path = os.path.join(ckpt_dir, f"{step}.pt")
                torch.save({
                    'config': cfg,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'step': step,
                }, ckpt_path)
        except Exception as e:
            print(f"Error at step {step}: {e}")
            # continue

if __name__ == "__main__":
    main()
