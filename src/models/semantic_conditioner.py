"""ESM-based semantic conditioner for UniChiFlow."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

try:
    import esm  # type: ignore
except Exception as exc:  # pragma: no cover
    esm = None
    _esm_import_error = exc
else:
    _esm_import_error = None

from src.data.twin_universe_dataset import tokens_to_strings


@dataclass
class ESM2Config:
    model: str = "esm2_t6_8M_UR50D"
    repr_layer: int = -1
    requires_grad: bool = False


class ESM2Conditioner(nn.Module):
    """Thin wrapper over `esm.pretrained` with frozen weights by default."""

    def __init__(self, cfg: Optional[ESM2Config] = None):
        super().__init__()
        self.cfg = cfg or ESM2Config()
        if esm is None or not hasattr(esm, "pretrained"):
            raise ImportError(
                "esm package is missing or too old. Install Meta ESM via `pip install fair-esm` "
                "(or `pip install git+https://github.com/facebookresearch/esm.git`). "
                f"Original import error: {_esm_import_error}"
            )
        try:
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.cfg.model)
        except Exception as exc:  # pragma: no cover - dependency issues
            raise ImportError(
                "Failed to load ESM weights. Ensure `fair-esm` is installed and model name is valid. "
                f"Tried model={self.cfg.model}. Underlying error: {exc}"
            ) from exc
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = bool(self.cfg.requires_grad)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.hidden_size = self.model.embed_dim
        self.repr_layer = self.cfg.repr_layer if self.cfg.repr_layer > 0 else self.model.num_layers

    @torch.no_grad()
    def forward(self, seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ESM embeddings for a batch of tokenized sequences.

        The ESM2 forward runs under an explicit fp32 autocast scope: pretrained ESM
        weights have large activations (especially attention softmax pre-scores)
        that overflow to Inf under fp16/bf16 autocast, poisoning downstream features
        and blowing up the flow loss. Casting the projected output back to the
        caller's AMP dtype keeps the rest of the graph in AMP.

        Args:
            seq: (B,L) int tokens in 0-39 (20 for PAD).
            mask: optional (B,L) bool mask (True=valid).
        Returns:
            embeddings: (B,L,D_esm)
            esm_mask: (B,L) bool mask used for ESM (excludes pads)
        """
        device = seq.device
        strings: List[Tuple[str, str]] = []
        B, L = seq.shape
        if mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=device)
        # convert to strings
        aa_strings = tokens_to_strings(seq.cpu(), mask.cpu())
        for i, s in enumerate(aa_strings):
            strings.append((f"seq_{i}", s))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(strings)
        batch_tokens = batch_tokens.to(device)

        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=False):
            with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.model.parameters())):
                out = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
            token_reps = out["representations"][self.repr_layer]  # (B, L+1, D)
            token_reps = token_reps[:, 1:, :]  # drop CLS

            # Trim to original length using mask lengths
            lengths = mask.sum(dim=1).tolist()
            rep_padded = torch.zeros((B, L, token_reps.size(-1)), device=device, dtype=token_reps.dtype)
            esm_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
            for i, length in enumerate(lengths):
                rep_padded[i, :length] = token_reps[i, :length]
                esm_mask[i, :length] = True

        return rep_padded, esm_mask
