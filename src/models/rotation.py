"""6D rotation representation (Zhou et al., 2019): first two columns of R <-> rotation matrix."""
import torch
import torch.nn.functional as F


def rotmat_to_6d(R: torch.Tensor) -> torch.Tensor:
    # (...,3,3) -> (...,6) : first two columns
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    # (...,6) -> (...,3,3) via Gram-Schmidt
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    # eps=1e-6 (not the default 1e-12): under fp16/AMP the head can emit near-zero
    # 6D vectors early in training, which F.normalize then turns into NaN when the
    # default eps underflows. Matches the Gram–Schmidt guard used in PFF.
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)
