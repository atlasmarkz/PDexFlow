import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple

from unichiflow.models.chiralformer import UniChiFlowModel
from unichiflow.models.hard_geometric_decoder import reconstruct_atom_positions
from unichiflow.flow import compute_symmetry_loss, target_velocity, predicted_velocity, flow_matching_loss
from unichiflow.data.pep_dataloader import TwinUniversePepDataset, PaddingCollate
from unichiflow import flow as ph_flow

class UniChiFlowSystem(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        
        # Initialize Model (Module 2: Chiralformer with Chiral-Bridge LoRA)
        self.model = UniChiFlowModel(
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
            angle_dim=int(cfg.model.get("angle_dim", 5)),
            use_ot=bool(cfg.flow.get("use_ot", True)),
        )

    def setup(self, stage: Optional[str] = None):
        # Module 1: Data Loading with Twin-Universe Strategy
        if stage == "fit" or stage is None:
            self.train_dataset = TwinUniversePepDataset(
                structure_dir=str(self.cfg.data.structure_dir),
                cache_dir=str(self.cfg.data.cache_dir),
                split_name=f"pepmerge_{self.cfg.data.split}",
                mirror_prob=float(self.cfg.data.twin_universe_prob),
                reset=bool(self.cfg.data.get("reset_cache", False)),
                deterministic_pairs=bool(self.cfg.data.get("deterministic_pairs", False)),
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.train.num_workers),
            collate_fn=PaddingCollate(eight=True),
            pin_memory=True,
        )

    def training_step(self, batch, batch_idx):
        # Forward pass handles corruption and prediction
        # The model's forward method already implements:
        # 1. Twin-Universe inputs (from batch 'chirality' token)
        # 2. Mini-ROT (Minibatch Optimal Transport) in 'corrupt_batch'
        # 3. Flow Matching Loss + Symmetry Consistency Loss
        loss_dict = self.model(batch)
        
        self.log('train/loss', loss_dict['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/cfm_loss', loss_dict['cfm_loss'], on_step=True, on_epoch=False)
        self.log('train/sym_loss', loss_dict['sym_loss'], on_step=True, on_epoch=False)
        self.log('train/seq_loss', loss_dict['seq_loss'], on_step=True, on_epoch=False)
        
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        # For validation, we can compute loss and also try to decode
        loss_dict = self.model(batch)
        self.log('val/loss', loss_dict['loss'], on_epoch=True)

        # Module 4: Geometric Decoder Check (Sampling & Reconstruction)
        # Taking the first sample in batch to demonstrate decoding
        if batch_idx == 0:
            with torch.no_grad():
                # Perform sampling
                sample_out = self.model.sample(batch, num_steps=10)
                
                # Use Hard Geometric Constraints Decoder to reconstruct atoms
                pos15, mask15 = reconstruct_atom_positions(
                    frames={
                        'aa': sample_out['seq'],
                        'trans': sample_out['trans'],
                        'rotmats': sample_out['rotmats']
                    },
                    torsions=sample_out['angles'],
                    chirality_token=batch['chirality'], # Use batch chirality for validation consistency
                    peptide_mask=batch['generate_mask']
                )
                # (Optional) Calculate some metric or just log validity
                self.log('val/valid_atoms', mask15.sum().float())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.train.optimizer.lr),
            weight_decay=float(self.cfg.train.optimizer.weight_decay)
        )
        return optimizer

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val=None, gradient_clip_algorithm=None):
        """Guard against exploding gradients without relying on AMP."""
        clip_val = float(self.cfg.train.get("grad_clip", 1.0))
        clip_algo = self.cfg.train.get("grad_clip_algorithm", "norm")
        self.clip_gradients(optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm=clip_algo)
