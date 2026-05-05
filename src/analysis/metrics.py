"""Evaluation metrics for UniChiFlow outputs.

Includes chirality error, clash score, RMSD (Kabsch), and amino-acid recovery.
"""
from typing import Optional, Tuple

import torch


def chirality_error(N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor, CB: torch.Tensor, chirality: torch.Tensor) -> torch.Tensor:
    """Compute fraction of residues violating target chirality.

    Args:
        N, CA, C, CB: (...,3) atom positions.
        chirality: (...,) 0 for L, 1 for D.
    Returns:
        Scalar tensor: mean violation rate.
    """
    v1 = N - CA
    v2 = C - CA
    v3 = CB - CA
    triple = torch.einsum('...i,...i->...', torch.cross(v1, v2, dim=-1), v3)
    expected_sign = torch.where(chirality.bool(), -1.0, 1.0)
    violations = (triple * expected_sign <= 0).float()
    return violations.mean()


def clash_score(pos: torch.Tensor, mask: Optional[torch.Tensor] = None, threshold: float = 1.5, bond_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Count atom pairs closer than threshold (excludes self and optional bonds)."""
    if mask is None:
        mask = torch.ones(pos.shape[:-1], device=pos.device, dtype=torch.bool)
    mask = mask.float()
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    dist = torch.linalg.norm(diff + 1e-9, dim=-1)
    pair_mask = (mask[:, :, None] * mask[:, None, :]).bool()
    eye = torch.eye(pair_mask.size(-1), device=pos.device, dtype=torch.bool)
    pair_mask = pair_mask & (~eye)
    if bond_mask is not None:
        pair_mask = pair_mask & (~bond_mask.bool())
    violations = (dist < threshold) & pair_mask
    counts = violations.sum(dim=[1, 2]).float() / 2.0  # symmetric pairs
    return counts.mean()


def kabsch_rmsd(X: torch.Tensor, Y: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    """RMSD with Kabsch alignment (batchable)."""
    if mask is None:
        mask = torch.ones(X.shape[:-1], device=X.device, dtype=torch.bool)
    mask = mask.float()
    w = mask / (mask.sum(dim=-1, keepdim=True).clamp(min=eps))
    Xc = X - (w[..., None] * X).sum(dim=-2, keepdim=True)
    Yc = Y - (w[..., None] * Y).sum(dim=-2, keepdim=True)
    cov = torch.matmul(Xc.transpose(-1, -2) * w[..., None, :], Yc)
    U, S, Vh = torch.linalg.svd(cov)
    R = torch.matmul(U, Vh)
    det = torch.det(R)
    fix = torch.eye(3, device=X.device, dtype=X.dtype)
    fix[-1, -1] = torch.where(det < 0, -1.0, 1.0)
    R = torch.matmul(U, torch.matmul(fix, Vh))
    X_aligned = torch.matmul(Xc, R)
    diff2 = (X_aligned - Yc).pow(2).sum(dim=-1)
    rmsd = torch.sqrt((diff2 * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=eps))
    return rmsd.mean()


def aar_accuracy(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Amino Acid Recovery accuracy.

    Args:
        pred: logits or int predictions (B,L,20) or (B,L).
        target: ground truth ints (B,L).
        mask: optional bool mask (B,L).
    Returns:
        Scalar accuracy.
    """
    if pred.dim() == target.dim() + 1:
        pred_idx = pred.argmax(dim=-1)
    else:
        pred_idx = pred
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    correct = (pred_idx == target) & mask
    return correct.float().sum() / mask.float().sum().clamp(min=1.0)
