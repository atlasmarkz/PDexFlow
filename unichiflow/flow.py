"""Parity-Harmonized Flow utilities and losses.

Implements PH-Flow on SE(3)^N x T^{kN} with parity consistency.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from unichiflow.data import so3, torus


# Prompt parity flip I_flip = diag(1,1,-1) (used for local-frame reflection).
# For global rotations we right-multiply by an SO(3) reflection-preserving factor
# I_SO3 = diag(-1,-1,1) so that det stays +1 while yielding x' = -(R @ x + t)
# once combined with the decoder's local flip.
PARITY_FLIP = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
PARITY_SO3 = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))


def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
	return torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim, keepdim=keepdim), min=eps))


def parity_positions(x: torch.Tensor) -> torch.Tensor:
	return -x


def parity_rotations(R: torch.Tensor) -> torch.Tensor:
	M = PARITY_SO3.to(device=R.device, dtype=R.dtype)
	# right-multiply to remain on SO(3) while matching Definition 1 with decoder flip
	return torch.matmul(R, M)


def parity_rotvec(omega: torch.Tensor) -> torch.Tensor:
	M = PARITY_SO3.to(device=omega.device, dtype=omega.dtype)
	return torch.matmul(omega, M)


def parity_torsions(theta: torch.Tensor) -> torch.Tensor:
	return -theta


def apply_parity_state(trans: torch.Tensor, rot: torch.Tensor, ang: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	return parity_positions(trans), parity_rotations(rot), parity_torsions(ang)


def target_velocity(trans_0: torch.Tensor, trans_1: torch.Tensor, rot_0: torch.Tensor, rot_1: torch.Tensor, ang_0: torch.Tensor, ang_1: torch.Tensor):
	v_trans = trans_1 - trans_0
	rel_rot = so3.relative_rotation(rot_0, rot_1)
	v_rot = so3.log_map(rel_rot)
	v_ang = torus.tor_logmap(ang_0, ang_1)
	return v_trans, v_rot, v_ang


def predicted_velocity(pred_trans_1: torch.Tensor, pred_rot_1: torch.Tensor, pred_ang_1: torch.Tensor, trans_t: torch.Tensor, rot_t: torch.Tensor, ang_t: torch.Tensor, t: torch.Tensor, eps: float = 1e-4):
	dt = torch.clamp(1.0 - t, min=eps)[..., None]
	v_trans = (pred_trans_1 - trans_t) / dt
	rot_delta = so3.relative_rotation(rot_t, pred_rot_1)
	v_rot = so3.log_map(rot_delta) / dt
	v_ang = torus.tor_logmap(ang_t, pred_ang_1) / dt
	return v_trans, v_rot, v_ang


def flow_matching_loss(pred: Tuple[torch.Tensor, ...], target: Tuple[torch.Tensor, ...], mask: torch.Tensor) -> torch.Tensor:
    """Compute loss directly on predicted state (x_1) vs target state (x_1).
    
    Args:
        pred: (trans, rot, ang) predicted for x_1
        target: (trans, rot, ang) ground truth x_1
    """
    losses = []
    # Translation: MSE
    if mask.any():
        losses.append(F.mse_loss(pred[0][mask], target[0][mask]))
    else:
        losses.append(pred[0].new_tensor(0.0))

    # Rotation: MSE on Matrices (Chordal distance)
    # This avoids the singularity of LogMap at 180 degrees and the 1/(1-t) explosion
    if mask.any():
        # Flatten matrices to (N, 9) for MSE, or just elementwise
        losses.append(F.mse_loss(pred[1][mask], target[1][mask]))
    else:
        losses.append(pred[1].new_tensor(0.0))

    # Torsion: MSE on (sin, cos) or raw angles?
    # Model predicts raw angles, but we usually want to account for periodicity.
    # However, target ang is in [0, 2pi].
    # Simple MSE is risky for periodicity. 
    # Better to use (sin, cos) distance or 1 - cos(delta).
    # Since predict_x1 outputs angles in [0, 2pi] via sigmoid/modulo?
    # The model `out_angles` has no activation in `chiralformer.py`, just Linear.
    # Then `pred_ang_1 = self.out_angles(h) % (2 * math.pi)`.
    # Modulo is bad for gradients at boundary.
    # Actually, let's look at `chiralformer.py` again.
    # `pred_ang_1 = self.out_angles(h) % (2 * math.pi)` -> This breaks gradient if it wraps?
    # No, it's just value wrapping.
    # BUT, if target is 0.1 and pred is 6.2, MSE is huge, but they are close.
    # We should use periodic loss: 1 - cos(pred - target).
    if mask.any():
        diff = pred[2][mask] - target[2][mask]
        # loss = 1 - cos(diff) = 2 * sin^2(diff/2)
        ang_loss = (1.0 - torch.cos(diff)).mean()
        losses.append(ang_loss)
    else:
        losses.append(pred[2].new_tensor(0.0))
        
    return sum(losses)


def symmetry_consistency_loss(pred_out: Tuple[torch.Tensor, ...], pred_sym_out: Tuple[torch.Tensor, ...], mask: torch.Tensor) -> torch.Tensor:
    """Enforce Parity(Predictions(x)) == Predictions(Parity(x))."""
    
    # Parity transform of the output from the mirrored input
    # We want pred_original approx Parity_Inverse(pred_sym)
    # Or Parity(pred_original) approx pred_sym
    
    # Calculate Parity(pred_original)
    p_trans = parity_positions(pred_out[0])
    p_rot = parity_rotations(pred_out[1])
    p_ang = parity_torsions(pred_out[2])
    
    losses = []
    
    if mask.any():
        losses.append(F.mse_loss(p_trans[mask], pred_sym_out[0][mask]))
        losses.append(F.mse_loss(p_rot[mask], pred_sym_out[1][mask])) # Matrix MSE
        
        diff = p_ang[mask] - pred_sym_out[2][mask]
        losses.append((1.0 - torch.cos(diff)).mean())
    else:
        losses.append(pred_out[0].new_tensor(0.0))
        
    return sum(losses)



def _clone_batch_with_chirality(batch: Dict[str, torch.Tensor], chirality: torch.Tensor) -> Dict[str, torch.Tensor]:
	"""Utility: shallow-copy a batch and replace chirality token."""
	batch_sym = {}
	for k, v in batch.items():
		if isinstance(v, torch.Tensor):
			batch_sym[k] = v.clone()
		else:
			batch_sym[k] = v
	batch_sym['chirality'] = chirality
	return batch_sym


def compute_symmetry_loss(predict_fn, batch: Dict[str, torch.Tensor], t: torch.Tensor, seq_t: torch.Tensor, trans_t: torch.Tensor, rot_t: torch.Tensor, ang_t: torch.Tensor, mask: torch.Tensor,
                             pred_out: Optional[Dict[str, torch.Tensor]] = None, v_pred: Optional[Tuple[torch.Tensor, ...]] = None) -> torch.Tensor:
    """Compute Eq.(3) symmetry consistency by re-evaluating the model on mirrored inputs.

    Args:
        pred_out: optional cached output from `predict_fn` to avoid recomputation.
        v_pred: IGNORED (kept for compatibility), we use states now.
    """
    if pred_out is None:
        pred_out = predict_fn(batch, t, seq_t, trans_t, rot_t, ang_t)
    
    # Extract prediction states
    pred_states = (pred_out['pred_trans_1'], pred_out['pred_rot_1'], pred_out['pred_ang_1'])

    chir = batch.get('chirality', torch.zeros((trans_t.size(0),), device=trans_t.device, dtype=torch.long))
    if isinstance(chir, torch.Tensor) and chir.ndim == 2 and chir.shape[-1] == 1:
        chir = chir.squeeze(-1)
    chir_mirror = torch.ones_like(chir) - chir

    batch_sym = _clone_batch_with_chirality(batch, chir_mirror)
    trans_p, rot_p, ang_p = apply_parity_state(trans_t, rot_t, ang_t)
    out_sym = predict_fn(batch_sym, t, seq_t, trans_p, rot_p, ang_p)
    
    pred_sym_states = (out_sym['pred_trans_1'], out_sym['pred_rot_1'], out_sym['pred_ang_1'])

    return symmetry_consistency_loss(pred_states, pred_sym_states, mask)


def riemannian_distance(trans_a: torch.Tensor, rot_a: torch.Tensor, ang_a: torch.Tensor, trans_b: torch.Tensor, rot_b: torch.Tensor, ang_b: torch.Tensor) -> torch.Tensor:
	d_trans = safe_norm(trans_a - trans_b, dim=-1)
	d_rot = so3.geodesic_distance(rot_a, rot_b)
	d_ang = safe_norm(torus.tor_logmap(ang_a, ang_b), dim=-1)
	return torch.sqrt(d_trans.square() + d_rot.square() + d_ang.square())
