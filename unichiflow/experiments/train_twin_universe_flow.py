"""Flow matching training with Twin-Universe dataset + ESM conditioner.

This script is a lightweight reference implementation of the blueprint described in the prompt:
- TwinUniverseDataset with on-the-fly chirality augmentation
- Minibatch OT alignment between Gaussian noise and data coordinates
- Frozen ESM conditioner feeding semantic context
- Chiral bridge gating (only active for D-mode samples)
- Two-stage training: full fine-tune (Stage 1) then LoRA-only (Stage 2)

Note: This module is intentionally compact and CPU-friendly; adjust configs as needed.
"""
import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from unichiflow.data.minibatch_ot import minibatch_ot_align
from unichiflow.data.twin_universe_dataset import TwinUniverseCollate, TwinUniverseDataset
from unichiflow.models.semantic_conditioner import ESM2Conditioner, ESM2Config
from unichiflow.models.chiral_gated_lora import wrap_transformer_with_chiral_lora


def time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1)))
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


@dataclass
class TrainConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    max_iters: int = 1000
    device: str = "cuda"
    vocab_size: int = 40  # 0-39 (L+D), 20 is PAD/MASK
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    mirror_prob: float = 0.5
    stage: int = 1  # 1 or 2
    log_every: int = 20


class ChiralFormerLite(nn.Module):
    def __init__(self, cfg: TrainConfig, esm_dim: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.seq_emb = nn.Embedding(cfg.vocab_size + 1, cfg.d_model)
        self.pos_emb = nn.Embedding(2048, cfg.d_model)
        self.chir_emb = nn.Embedding(2, cfg.d_model)
        self.coord_in = nn.Linear(3, cfg.d_model)
        self.t_in = nn.Linear(cfg.d_model, cfg.d_model)
        self.esm_proj = nn.Linear(esm_dim, cfg.d_model)
        self.bridge_scale = nn.Parameter(torch.zeros(1))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        if cfg.use_lora:
            self._chiral_lora_modules = wrap_transformer_with_chiral_lora(self.trunk, rank=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
        else:
            self._chiral_lora_modules = tuple()

        self.head_vel = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 3))
        self.head_seq = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.vocab_size))

    def forward(self, x_t: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor, chirality: torch.Tensor, t: torch.Tensor, esm_cond: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, L, _ = x_t.shape
        pos_ids = torch.arange(L, device=x_t.device)[None, :].repeat(B, 1)
        x = self.seq_emb(seq.clamp(min=0, max=self.cfg.vocab_size))
        x = x + self.pos_emb(pos_ids)
        x = x + self.chir_emb(chirality.view(B, 1).expand(-1, L))
        x = x + self.coord_in(x_t)
        x = x + self.t_in(time_embed(t.view(B), self.d_model))[:, None, :]

        if esm_cond is not None:
            if esm_cond.shape[:2] != (B, L):
                raise ValueError(f"esm_cond shape mismatch: {esm_cond.shape} vs (B={B},L={L})")
            x = x + torch.sigmoid(self.bridge_scale) * esm_cond * chirality.view(B, 1, 1).to(esm_cond.dtype)

        if self._chiral_lora_modules:
            gate = chirality.view(B, 1, 1).to(x.dtype)
            for m in self._chiral_lora_modules:
                m.set_gate(gate)
        h = self.trunk(x, src_key_padding_mask=~mask)
        if self._chiral_lora_modules:
            for m in self._chiral_lora_modules:
                m.set_gate(None)
        v = self.head_vel(h)
        logits = self.head_seq(h)
        return {"vel": v, "logits": logits}


def freeze_backbone_for_stage2(model: ChiralFormerLite):
    for name, param in model.named_parameters():
        if "lora_" in name or "bridge_scale" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def run_train(args: argparse.Namespace):
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = TrainConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        device=device,
        vocab_size=40,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        mirror_prob=args.mirror_prob,
        stage=args.stage,
        log_every=args.log_every,
    )

    with open(args.pdb_list, 'r') as f:
        pdb_paths = [line.strip() for line in f if line.strip()]
    ds = TwinUniverseDataset(pdb_paths=pdb_paths, mirror_prob=cfg.mirror_prob)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=args.num_workers,
                        collate_fn=TwinUniverseCollate(pad_to_multiple_of=8), drop_last=True)
    def _inf_iter(dataloader):
        while True:
            for batch in dataloader:
                yield batch
    it = _inf_iter(loader)

    # conditioners & model
    esm_cond = ESM2Conditioner(ESM2Config(model=args.esm_model, repr_layer=args.esm_layer, requires_grad=False)).to(device)
    model = ChiralFormerLite(cfg, esm_dim=esm_cond.hidden_size).to(device)

    if cfg.stage == 2:
        freeze_backbone_for_stage2(model)

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pbar = tqdm(range(1, cfg.max_iters + 1))
    for step in pbar:
        batch = next(it)
        coords = batch['coords'].to(device)  # (B,L,3)
        seq = batch['seq'].to(device)
        mask = batch['mask'].to(device)
        chirality = batch['chirality_token'].to(device)

        # OT alignment between Gaussian noise and data coords
        x1 = coords
        x0 = torch.randn_like(x1)
        x0_aligned, _ = minibatch_ot_align(x0, x1, mask)

        # Flow interpolation
        t = torch.rand((coords.size(0),), device=device)
        t_broadcast = t.view(-1, 1, 1)
        x_t = (1 - t_broadcast) * x0_aligned + t_broadcast * x1

        with torch.no_grad():
            esm_embed, _ = esm_cond(seq, mask)
            esm_embed = esm_embed.to(device)

        out = model(x_t, seq, mask, chirality, t, esm_embed)
        v_target = x1 - x0_aligned
        vel_loss = F.mse_loss(out['vel'][mask], v_target[mask])
        seq_loss = F.cross_entropy(out['logits'][mask], seq[mask].clamp(max=cfg.vocab_size - 1))
        loss = vel_loss + seq_loss

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % cfg.log_every == 0:
            pbar.set_description(f"step {step} loss {loss.item():.4f} vel {vel_loss.item():.4f} seq {seq_loss.item():.4f}")

    print("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twin-Universe flow matching training")
    parser.add_argument("pdb_list", type=str, help="Path to a text file containing PDB paths (one per line)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--mirror_prob", type=float, default=0.5)
    parser.add_argument("--esm_model", type=str, default="esm2_t6_8M_UR50D")
    parser.add_argument("--esm_layer", type=int, default=-1)
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA injection")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()
    run_train(args)
