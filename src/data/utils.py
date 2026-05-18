import math
import os
import random
from typing import Any, Dict

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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
