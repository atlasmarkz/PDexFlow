import os
import sys
import time
import hydra
import torch
from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path if package is not installed
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from unichiflow.data.pep_dataloader import PaddingCollate
from unichiflow.data.real_d_pep_dataset import RealDPepDataset
from unichiflow.data import utils as du
from unichiflow.models.chiralformer import UniChiFlowModel
from unichiflow.checkpoint import load_backbone_into_lora

config_path = os.path.join(os.path.dirname(__file__), "..", "configs")
config_name = "unichiflow_lora.yaml"

def _inf_iterator(loader):
    while True:
        for batch in loader:
            yield batch

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

    # Use RealDPepDataset
    ds = RealDPepDataset(
        structure_dir=str(cfg.data.structure_dir),
        cache_dir=str(cfg.data.cache_dir),
        name=f"real_d_{cfg.data.split}",
        reset=bool(cfg.data.reset_cache),
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
        esm=cfg.get("esm"),  # Pass ESM config if needed
        seq_weight=float(cfg.flow.get("seq_weight", 1.0)),
        angle_dim=int(cfg.model.get("angle_dim", 5)),
        use_ot=bool(cfg.flow.get("use_ot", True)),
    ).to(device)

    # Load pre-trained backbone
    if cfg.sample.ckpt_path:
        print(f"Loading pre-trained backbone from {cfg.sample.ckpt_path}...")
        ckpt = torch.load(str(cfg.sample.ckpt_path), map_location=device, weights_only=False)
        
        # Use intelligent loader that handles LoRA key mismatches
        # This handles mapping 'weight' -> 'base.weight' if the model has LoRA layers
        model = load_backbone_into_lora(model, ckpt['model'])
    else:
        print("WARNING: No checkpoint path provided. Training from scratch (random initialization)!")

    # If freeze_backbone is enabled, only train LoRA parameters
    if cfg.chi_lora.enable and cfg.chi_lora.freeze_backbone:
        print("Freezing backbone, training only LoRA parameters.")
        # The model handles freezing in _apply_chi_lora if configured, 
        # but let's double check or ensure optimizer only sees trainable params.
        # UniChiFlowModel._apply_chi_lora calls freeze_backbone if configured.
        pass
    
    # Filter parameters that require grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")

    optim = torch.optim.AdamW(trainable_params, lr=float(cfg.train.optimizer.lr), weight_decay=float(cfg.train.optimizer.weight_decay))
    
    max_grad_value = float(cfg.train.get("max_grad_value", 5.0))

    def _sanitize_gradients(parameters):
        """Replace NaN/Inf grads and optionally clip by value before norm clipping."""
        for p in parameters:
            if p.grad is None:
                continue
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if max_grad_value > 0:
                p.grad.clamp_(min=-max_grad_value, max=max_grad_value)

    model.train()
    pbar = tqdm(range(1, int(cfg.train.max_iters) + 1))
    
    best_loss = float('inf')
    
    for step in pbar:
        try:
            batch = du.recursive_to(next(it), device)
            
            # Normalize coordinates to improve numerical stability
            if 'pos_heavyatom' in batch:
                batch['pos_heavyatom'] = batch['pos_heavyatom'] * 0.1

            loss_dict = model(batch)
            loss = loss_dict['loss']

            # Skip step if any loss component is non-finite
            if not torch.isfinite(loss):
                print(f"Non-finite loss at step {step}: {loss_dict}")
                optim.zero_grad(set_to_none=True)
                continue
            
            optim.zero_grad(set_to_none=True)
            loss.backward()
            
            _sanitize_gradients(trainable_params)
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
