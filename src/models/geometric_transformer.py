"""Target-conditioned Geometric Transformer (paper Sec. IV-C).

* Pairwise CA distances are encoded with radial basis functions (RBF) and injected as per-head
  attention biases. Distances are parity-invariant, so chirality sensitivity comes from the frame /
  torsion features and the chirality embedding in the input featurization (Eq. 9).
* Every block alternates peptide self-attention and receptor cross-attention, followed by an FFN
  (pre-LayerNorm throughout).
* q/k/v/out projections are separate ``nn.Linear`` modules (and ``linear1``/``linear2`` for the FFN),
  so Chiral-Bridge LoRA can wrap any of them.

Token groups are given by ``pep_mask`` (generated peptide residues) and ``rec_mask`` (receptor
context). Self-attention only mixes tokens inside the same group; cross-attention lets peptide
queries attend to receptor keys/values. Receptor tokens are fixed context and receive no cross-attn update.
"""
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBF(nn.Module):
    """Gaussian radial basis expansion of distances given in angstrom."""

    def __init__(self, n_rbf: int = 16, d_min: float = 0.0, d_max: float = 20.0):
        super().__init__()
        centers = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer("centers", centers, persistent=False)
        self.gamma = 1.0 / (2.0 * ((d_max - d_min) / max(n_rbf - 1, 1)) ** 2)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers.to(d.dtype)) ** 2)


class BiasedAttention(nn.Module):
    """Multi-head attention with separate q/k/v/out projections and an additive per-head pair bias."""

    def __init__(self, d_model: int, n_heads: int, n_rbf: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.pair_bias = nn.Linear(n_rbf, n_heads, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, allowed: torch.Tensor, rbf: torch.Tensor) -> torch.Tensor:
        """x_q, x_kv: (B,L,D); allowed: (B,L,L) bool, True where query i may attend key j; rbf: (B,L,L,n_rbf)."""
        B, L, _ = x_q.shape
        H, Dh = self.n_heads, self.d_head
        q = self.q_proj(x_q).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(x_kv).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(x_kv).view(B, L, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)
        scores = scores + self.pair_bias(rbf).permute(0, 3, 1, 2).to(scores.dtype)

        # Rows without any allowed key (e.g. receptor queries in cross-attention, or a complex
        # with no receptor): keep softmax finite, then zero their output.
        has_key = allowed.any(dim=-1)  # (B,L)
        safe_allowed = allowed.clone()
        safe_allowed[..., 0] = safe_allowed[..., 0] | (~has_key)
        scores = scores.masked_fill(~safe_allowed.unsqueeze(1), float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, H * Dh)
        out = self.out_proj(out)
        return out * has_key.unsqueeze(-1).to(out.dtype)


class GeometricTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_rbf: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = BiasedAttention(d_model, n_heads, n_rbf, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = BiasedAttention(d_model, n_heads, n_rbf, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, self_allowed, cross_allowed, rbf):
        n1 = self.norm1(h)
        h = h + self.dropout(self.self_attn(n1, n1, self_allowed, rbf))
        n2 = self.norm2(h)
        h = h + self.dropout(self.cross_attn(n2, n2, cross_allowed, rbf))
        n3 = self.norm3(h)
        h = h + self.dropout(self.linear2(self.dropout(F.gelu(self.linear1(n3)))))
        return h


class GeometricTransformer(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float = 0.1, n_rbf: int = 16, rbf_max: float = 20.0):
        super().__init__()
        self.rbf = RBF(n_rbf=n_rbf, d_max=rbf_max)
        self.layers = nn.ModuleList([GeometricTransformerLayer(d_model, n_heads, n_rbf, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,
        ca_angstrom: torch.Tensor,
        res_mask: torch.Tensor,
        pep_mask: torch.Tensor,
        pre_layer_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """h: (B,L,D); ca_angstrom: (B,L,3) CA positions in angstrom; masks: (B,L) bool."""
        res_mask = res_mask.bool()
        pep = pep_mask.bool() & res_mask
        rec = (~pep_mask.bool()) & res_mask
        same_group = (pep.unsqueeze(2) & pep.unsqueeze(1)) | (rec.unsqueeze(2) & rec.unsqueeze(1))
        self_allowed = same_group
        cross_allowed = pep.unsqueeze(2) & rec.unsqueeze(1)

        d = torch.cdist(ca_angstrom.float(), ca_angstrom.float()).to(h.dtype)
        rbf = self.rbf(d)

        for i, layer in enumerate(self.layers):
            if pre_layer_hook is not None:
                h = pre_layer_hook(i, h)
            h = layer(h, self_allowed, cross_allowed, rbf)
        return self.final_norm(h)
