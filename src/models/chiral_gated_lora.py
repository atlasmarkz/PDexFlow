"""Chiral-gated LoRA layers for UniChiFlow."""
import math
from typing import Optional

import torch
import torch.nn as nn


class ChiralBridgeLoRA(nn.Module):
    """LoRA adapter with a chirality gate (Chiral-Bridge).

    Applies LoRA only when `chirality_token==1` (D-mode) and scales with a learnable logit.
    Formula: h' = h + G(c) * alpha * (W_up @ W_down) * h
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive for ChiralBridgeLoRA")

        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros((self.rank, base.in_features))) # W_down
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, self.rank))) # W_up
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        self._gate: Optional[torch.Tensor] = None

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def set_gate(self, gate: Optional[torch.Tensor]):
        # gate is broadcastable to (B,1,1) or (B,L,1)
        self._gate = gate

    def forward(self, x: torch.Tensor, chirality_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        
        gate = None
        if self._gate is not None:
            gate = self._gate
        elif chirality_token is not None:
            # c=1: allow flow, c=0: block
            gate = chirality_token.view(x.shape[0], *([1] * (x.ndim - 1))).to(x.dtype)
        
        if gate is None:
            gate = 0.0
            
        # G(c) logic: learnable gate enabled by chirality token
        scale = torch.sigmoid(self.gate_logit) * gate
        return base_out + self.scaling * lora_out * scale


def wrap_transformer_with_chiral_lora(transformer: nn.TransformerEncoder, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0):
    """In-place wrap of a TransformerEncoder's FFN and attention output with ChiralGatedLoRA.

    Returns a tuple of the inserted LoRA modules so that callers can set gates per forward.
    """
    modules = []
    for layer in transformer.layers:
        layer.linear1 = ChiralGatedLoRA(layer.linear1, rank=rank, alpha=alpha, dropout=dropout)
        layer.linear2 = ChiralGatedLoRA(layer.linear2, rank=rank, alpha=alpha, dropout=dropout)
        layer.self_attn.out_proj = ChiralGatedLoRA(layer.self_attn.out_proj, rank=rank, alpha=alpha, dropout=dropout)
        modules.extend([layer.linear1, layer.linear2, layer.self_attn.out_proj])
    transformer._chiral_lora_modules = tuple(modules)
    return transformer._chiral_lora_modules
