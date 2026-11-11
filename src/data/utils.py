import math
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

MASK_TOKEN_INDEX = 20
NUM_AA_TOKENS = 20


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def recursive_to(obj: Any, device: str):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: recursive_to(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_to(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_to(v, device) for v in obj)
    return obj


def get_time_embedding(timesteps: torch.Tensor, embedding_dim: int, max_positions: int = 2000):
    assert timesteps.ndim == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1), mode='constant')
    return emb


def tokens_to_string(seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> str:
    """Convert integer tokens (0-39, 20 for pad) to an amino-acid string for ESM."""
    from src.data.pep_constants import resindex_to_ressymb  # lazy: pep_constants is heavy and not needed elsewhere in utils

    valid_len = int(seq.numel()) if mask is None else int(mask.sum().item())
    seq_l = (seq.view(-1)[:valid_len] % 20).tolist()
    return ''.join(resindex_to_ressymb[int(t)] for t in seq_l)


def tokens_to_strings(batch_seq: torch.Tensor, batch_mask: Optional[torch.Tensor] = None) -> List[str]:
    return [tokens_to_string(batch_seq[i], batch_mask[i] if batch_mask is not None else None) for i in range(batch_seq.shape[0])]
