"""Chiral-gated LoRA layers for PDexFlow."""
import math
from typing import Optional

import torch
import torch.nn as nn


class ChiralBridgeLoRA(nn.Module):
    """LoRA adapter with a chirality gate (Chiral-Bridge).

    Applies LoRA only when `chirality_token==1` (D-mode) and scales with a learnable logit.
    Formula: h' = h + G(c) * alpha * (W_up @ W_down) * h

    The learnable gate logit is initialized to give sigmoid(gamma) ≈ 0.5, allowing
    the network to initially behave as a vanilla transformer while gradually
    enabling D-mode adaptations. The chirality token (c) acts as a binary switch
    that can completely disable LoRA contributions for L-peptides (c=0).
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0, dropout: float = 0.0, gate_init: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive for ChiralBridgeLoRA")

        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros((self.rank, base.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, self.rank)))
        self.gate_logit = nn.Parameter(torch.tensor(gate_init))
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
        self._gate = gate

    def forward(self, x: torch.Tensor, chirality_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()

        gate = None
        if self._gate is not None:
            gate = self._gate
        elif chirality_token is not None:
            gate = chirality_token.view(x.shape[0], *([1] * (x.ndim - 1))).to(x.dtype)

        if gate is None:
            gate = 0.0

        gate_strength = torch.sigmoid(self.gate_logit)
        scale = gate_strength * gate
        return base_out + self.scaling * lora_out * scale
