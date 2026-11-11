"""Parity (spatial inversion) operator P used throughout PDexFlow.

Paper (Sec. III-C): for a residue frame (R, x),

    P(R, x) = (-R @ I_flip, -x),   I_flip = diag(1, 1, -1)

Positions are inverted (x -> -x, det = -1, a true mirror image). A backbone frame
built by Gram-Schmidt from (N, CA, C) of the inverted structure equals
``R @ diag(-1, -1, 1)`` (= ``-R @ I_flip``, det = +1), so the frame action is a
*right* multiplication by ``PARITY_SO3 = diag(-1, -1, 1)``. Dihedral angles change
sign under inversion.

Tangent-space actions (used by the symmetry consistency loss), derived from
``rel = R_from^T @ R_to``:
    translation velocity  v -> -v
    rotation velocity     w -> PARITY_SO3 @ w = (-wx, -wy, wz)   (conjugation of rel by PARITY_SO3)
    torsion velocity      u -> -u
All maps are involutions (P^-1 = P).
"""
import math

import torch

# Frame action: -I_flip, i.e. diag(-1, -1, 1). A proper rotation (det = +1).
PARITY_SO3 = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
# Local reflection used by the hard geometric decoder (paper Eq. 11): diag(1, 1, -1). det = -1.
LOCAL_REFLECTION = torch.diag(torch.tensor([1.0, 1.0, -1.0]))


def parity_positions(x: torch.Tensor) -> torch.Tensor:
    """Spatial inversion of coordinates / translations (or translation velocities)."""
    return -x


def parity_rotations(R: torch.Tensor) -> torch.Tensor:
    """Parity action on backbone frames: R -> R @ diag(-1, -1, 1)."""
    return R @ PARITY_SO3.to(device=R.device, dtype=R.dtype)


def parity_rotvec(w: torch.Tensor) -> torch.Tensor:
    """Parity action on rotation tangent vectors (relative-rotation convention of so3.relative_rotation)."""
    sign = torch.tensor([-1.0, -1.0, 1.0], device=w.device, dtype=w.dtype)
    return w * sign


def parity_torsions(a: torch.Tensor) -> torch.Tensor:
    """Dihedral angles flip sign under inversion; wrapped to [0, 2*pi) like the data pipeline."""
    return torch.remainder(-a, 2 * math.pi)


def parity_torsion_velocity(u: torch.Tensor) -> torch.Tensor:
    return -u


def apply_parity_state(trans: torch.Tensor, rot: torch.Tensor, ang: torch.Tensor):
    """Apply P to a peptide state (translations, frames, torsions)."""
    return parity_positions(trans), parity_rotations(rot), parity_torsions(ang)
