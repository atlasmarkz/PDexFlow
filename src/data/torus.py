import math
import torch


def tor_expmap(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return (x + u) % (2 * math.pi)


def tor_logmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # shortest signed difference on circle
    return torch.atan2(torch.sin(y - x), torch.cos(y - x))


def tor_projx(x: torch.Tensor) -> torch.Tensor:
    return x % (2 * math.pi)


def tor_random_uniform(*size, dtype=None, device=None) -> torch.Tensor:
    z = torch.rand(*size, dtype=dtype, device=device)
    return z * 2 * math.pi


def tor_geodesic_t(t: torch.Tensor, angles_1: torch.Tensor, angles_0: torch.Tensor) -> torch.Tensor:
    tangent_vec = t * tor_logmap(angles_0, angles_1)
    return tor_expmap(angles_0, tangent_vec)
