"""Minibatch Riemannian OT utilities for UniChiFlow."""
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Union

import torch

from src.data import so3, torus

try:  # optional SciPy dependency
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback when SciPy is absent
    linear_sum_assignment = None


# scipy.optimize.linear_sum_assignment is a C extension that releases the GIL,
# so running batch elements in a thread pool is a real wall-clock win for B>1.
# Capped modestly — the per-call work is small and too many threads only adds
# dispatch overhead. Reused across calls to avoid startup cost per step.
_OT_THREAD_POOL: Optional[ThreadPoolExecutor] = None
_OT_MAX_WORKERS = 8


def _get_ot_pool() -> ThreadPoolExecutor:
    global _OT_THREAD_POOL
    if _OT_THREAD_POOL is None:
        _OT_THREAD_POOL = ThreadPoolExecutor(max_workers=_OT_MAX_WORKERS, thread_name_prefix="ot")
    return _OT_THREAD_POOL


def pairwise_cost(z0: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
               z1: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
               mask1: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute Mini-ROT cost C_ij = d_M(z0(i), z1(j))^2.

    Supports two modes:
      - Euclidean (legacy): z0/z1 are tensors (B,N,3).
      - Riemannian (PH-Flow): z0/z1 are tuples (trans, rot, ang).
    """
    if isinstance(z0, tuple) or isinstance(z1, tuple):
        if not (isinstance(z0, tuple) and isinstance(z1, tuple) and len(z0) == 3 and len(z1) == 3):
            raise ValueError("For Riemannian cost, z0 and z1 must be (trans, rot, ang) tuples")
        trans0, rot0, ang0 = z0
        trans1, rot1, ang1 = z1
        if trans0.shape != trans1.shape:
            raise ValueError(f"trans shapes must match, got {tuple(trans0.shape)} vs {tuple(trans1.shape)}")
        B, N, _ = trans0.shape
        # broadcast pairwise distances
        d_trans = (trans0[:, :, None, :] - trans1[:, None, :, :]).square().sum(-1)
        d_rot = so3.geodesic_distance(rot0[:, :, None, :, :], rot1[:, None, :, :, :]).square()
        d_ang = torus.tor_logmap(ang0[:, :, None, :], ang1[:, None, :, :]).square().sum(-1)
        cost = d_trans + d_rot + d_ang
    else:
        if z0.shape != z1.shape:
            raise ValueError(f"x0 and x1 must have the same shape, got {tuple(z0.shape)} vs {tuple(z1.shape)}")
        B, N, _ = z0.shape
        diff = z0[:, :, None, :] - z1[:, None, :, :]
        cost = diff.square().sum(-1)

    if mask1 is not None:
        mask1 = mask1.to(cost.dtype)
        cost = cost + (1.0 - mask1[:, None, :]) * 1e6
    return cost


def _greedy_permutation(cost_b: torch.Tensor) -> torch.Tensor:
    """Greedy fallback that produces a valid permutation (no repeated columns).

    Worse than Hungarian but keeps the OT invariant: every source index maps to a
    unique target. Used only when SciPy is unavailable.
    """
    N = cost_b.shape[0]
    perm = torch.empty(N, dtype=torch.long, device=cost_b.device)
    taken = torch.zeros(N, dtype=torch.bool, device=cost_b.device)
    # Greedy row-by-row: for each source row, pick the cheapest unused target.
    work = cost_b.clone()
    for i in range(N):
        # mask out taken columns with +inf so they won't be picked
        j = int(torch.argmin(work[i]).item())
        perm[i] = j
        taken[j] = True
        work[:, j] = float('inf')
    return perm


def solve_ot(cost: torch.Tensor) -> torch.Tensor:
    """Solve linear assignment for each batch element.

    Returns permutation indices for x0 such that x0[perm] aligns to x1.
    """
    B, N, _ = cost.shape
    if linear_sum_assignment is None:
        # Keep a valid permutation when SciPy is missing. argmin per row would
        # produce repeated indices, which breaks gather-based alignment downstream.
        perms = [_greedy_permutation(cost[b].detach()) for b in range(B)]
        return torch.stack(perms, dim=0).to(cost.device)

    cost_cpu = cost.detach().cpu()
    cost_np = [cost_cpu[b].numpy() for b in range(B)]

    def _solve_one(mat):
        row_ind, col_ind = linear_sum_assignment(mat)
        return row_ind, col_ind

    # Small batches aren't worth the dispatch cost — just run inline.
    if B <= 2:
        results = [_solve_one(m) for m in cost_np]
    else:
        pool = _get_ot_pool()
        results = list(pool.map(_solve_one, cost_np))

    perms = []
    for row_ind, col_ind in results:
        if len(row_ind) != N:
            raise RuntimeError("OT solver returned incomplete assignment")
        p = torch.empty(N, dtype=torch.long)
        p[row_ind] = torch.as_tensor(col_ind, dtype=torch.long)
        perms.append(p)
    perm = torch.stack(perms, dim=0).to(cost.device)
    return perm


def minibatch_ot_align(x0: torch.Tensor, x1: torch.Tensor, mask1: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align noisy samples to data samples with OT permutation.

    Args:
        x0: (B,N,3) noisy Gaussian samples.
        x1: (B,N,3) data samples.
        mask1: optional (B,N) valid mask.
    Returns:
        x0_perm: permuted x0 that best matches x1.
        perm: permutation indices (B,N).
    """
    if isinstance(x0, tuple) or isinstance(x1, tuple):
        z0_perm, perm = minibatch_optimal_transport(x0, x1, mask1)
        return z0_perm, perm

    C = pairwise_cost(x0, x1, mask1)
    perm = solve_ot(C)
    # gather according to perm for each batch
    B, N, _ = x0.shape
    batch_idx = torch.arange(B, device=x0.device)[:, None].expand(-1, N)
    x0_perm = x0[batch_idx, perm]
    return x0_perm, perm


def minibatch_optimal_transport(z0: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                           z1: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                           mask: Optional[torch.Tensor] = None) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Hungarian-based OT alignment wrapper matching prompt nomenclature.

    Accepts either Euclidean tensors or (trans, rot, ang) tuples.
    """
    C = pairwise_cost(z0, z1, mask)
    perm = solve_ot(C)

    if isinstance(z0, tuple):
        trans0, rot0, ang0 = z0
        B, N, _ = trans0.shape
        batch_idx = torch.arange(B, device=trans0.device)[:, None].expand(-1, N)
        z0_perm = (
            trans0[batch_idx, perm],
            rot0[batch_idx, perm],
            ang0[batch_idx, perm],
        )
    else:
        B, N, _ = z0.shape
        batch_idx = torch.arange(B, device=z0.device)[:, None].expand(-1, N)
        z0_perm = z0[batch_idx, perm]
    return z0_perm, perm
