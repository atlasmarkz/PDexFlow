"""Minibatch Riemannian OT utilities for PDexFlow."""
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


def minibatch_ot_pair(z0, z1, pep_mask: torch.Tensor):
    """Mini-ROT of the paper (Eq. 13): optimal assignment between the B prior samples and B data samples.

        pi* = argmin_{pi in Pi_B} sum_i d_M^2(z0_i, z1_{pi(i)})

    d_M^2 is the product-manifold squared distance (translation + SO(3) geodesic + torus), averaged over the
    data sample's peptide residues. Prior samples are i.i.d., so slot l of prior i is compared with the l-th
    peptide residue of data sample j. The cost is parity-invariant, as required by Theorem 1.

    Args:
        z0, z1: tuples (trans (B,L,3), rot (B,L,3,3), ang (B,L,A)) of prior / data states.
        pep_mask: (B,L) bool, generated peptide residues.
    Returns:
        z0 re-assigned so that z0_out[j] is the prior matched to data sample j (peptide residues only;
        other positions are untouched).
    """
    trans0, rot0, ang0 = z0
    trans1, rot1, ang1 = z1
    B, L, _ = trans0.shape
    n_per = pep_mask.sum(dim=1)
    n_max = int(n_per.max().item()) if B > 0 else 0
    if B < 2 or n_max == 0:
        return z0

    # Gather peptide residues to the front: idx (B, n_max) = peptide positions, padded with non-peptide positions.
    order = torch.argsort((~pep_mask).to(torch.int8), dim=1, stable=True)[:, :n_max]
    valid = torch.arange(n_max, device=pep_mask.device)[None, :] < n_per[:, None]  # (B, n_max)

    def g(x):
        idx = order.view(B, n_max, *([1] * (x.dim() - 2))).expand(B, n_max, *x.shape[2:])
        return torch.gather(x, 1, idx)

    t0, r0, a0 = g(trans0), g(rot0), g(ang0)
    t1, r1, a1 = g(trans1), g(rot1), g(ang1)

    d_t = (t0[:, None] - t1[None]).square().sum(-1)                                   # (B_prior, B_data, n_max)
    d_r = so3.geodesic_distance(r0[:, None], r1[None]).square()
    d_a = torus.tor_logmap(a0[:, None], a1[None]).square().sum(-1)
    w = valid[None].to(d_t.dtype)                                                       # weights by data sample j
    cost = ((d_t + d_r + d_a) * w).sum(-1) / n_per[None, :].clamp(min=1).to(d_t.dtype)  # (B_prior, B_data)

    perm = solve_ot(cost[None])[0]            # perm[i] = data index matched to prior i
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(B, device=perm.device)  # inv[j] = prior index matched to data j

    def put(full, gathered_aligned):
        idx = order.view(B, n_max, *([1] * (full.dim() - 2))).expand(B, n_max, *full.shape[2:])
        vm = valid.view(B, n_max, *([1] * (full.dim() - 2)))
        merged = torch.where(vm, gathered_aligned, torch.gather(full, 1, idx))
        return full.scatter(1, idx, merged)

    return (put(trans0, t0[inv]), put(rot0, r0[inv]), put(ang0, a0[inv]))
