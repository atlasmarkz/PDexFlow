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


def target_velocity(trans_0: torch.Tensor, trans_1: torch.Tensor, rot_0: torch.Tensor, rot_1: torch.Tensor, ang_0: torch.Tensor, ang_1: torch.Tensor, sigma_min: float = 1e-4):
    v_trans = trans_1 - (1 - sigma_min) * trans_0
    rel_rot = so3.relative_rotation(rot_0, rot_1)
    v_rot = so3.log_map(rel_rot)
    v_ang = torus.tor_logmap(ang_0, ang_1)
    return v_trans, v_rot, v_ang


def predicted_velocity(pred_trans_1: torch.Tensor, pred_rot_1: torch.Tensor, pred_ang_1: torch.Tensor, trans_t: torch.Tensor, rot_t: torch.Tensor, ang_t: torch.Tensor, t: torch.Tensor, sigma_min: float = 1e-4):
    # Paper Eq 1 interpolation: z_t = (1 - (1-sigma)t)z_0 + t z_1
    # Implies v_t = (z_1 - (1-sigma)z_t) / (1 - (1-sigma)t) for vector space (translations)
    
    # Denominator for translations (avoid div by zero via sigma_min)
    # 1 - (1 - sigma) * t = 1 - t + sigma*t
    denom = 1.0 - (1.0 - sigma_min) * t
    denom = torch.clamp(denom, min=sigma_min).unsqueeze(-1) # (B, 1) broadcastable

    v_trans = (pred_trans_1 - (1.0 - sigma_min) * trans_t) / denom
    
    # For Rot/Ang on manifold, we assume geodesic flow approx (sigma_min ~ 0 for geometric paths)
    # Using simple 1-t for dt approx to match standard geodesic flow derivative
    dt = torch.clamp(1.0 - t, min=sigma_min)[..., None]
    
    rot_delta = so3.relative_rotation(rot_t, pred_rot_1)
    v_rot = so3.log_map(rot_delta) / dt
    v_ang = torus.tor_logmap(ang_t, pred_ang_1) / dt
    return v_trans, v_rot, v_ang


def flow_matching_loss(pred_x1: Tuple[torch.Tensor, ...], target_x1: Tuple[torch.Tensor, ...],
                       trans_t: torch.Tensor, rot_t: torch.Tensor, ang_t: torch.Tensor, t: torch.Tensor,
                       trans_0: torch.Tensor, rot_0: torch.Tensor, ang_0: torch.Tensor,
                       trans_1: torch.Tensor, rot_1: torch.Tensor, ang_1: torch.Tensor,
                       mask: torch.Tensor,
                       weights: Optional[Tuple[float, float, float]] = None,
                       huber_beta: float = 0.25,
                       v_clip: float = 50.0,
                       sigma_min: float = 1e-4) -> torch.Tensor:
    """Velocity-based flow matching loss (paper-aligned).
    
    Implements Eq 2 from UniDexFlow paper:
       L_flow = || v_theta - (z_1 - (1 - sigma_min)z_0) ||^2
    """
    if weights is None:
        w_trans, w_rot, w_ang = 1.0, 1.0, 1.0
    else:
        w_trans, w_rot, w_ang = float(weights[0]), float(weights[1]), float(weights[2])

    # compute target and predicted velocities
    v_true = target_velocity(trans_0, trans_1, rot_0, rot_1, ang_0, ang_1, sigma_min=sigma_min)
    v_pred = predicted_velocity(pred_x1[0], pred_x1[1], pred_x1[2], trans_t, rot_t, ang_t, t, sigma_min=sigma_min)

    # Stabilize the loss by down-weighting samples with small (1-t)?
    # Actually, with sigma_min, the velocity is finite. 
    # However, training can still be unstable if we don't weight or clip.
    # We keep the Huber loss and clipping for stability as requested by previous debug sessions,
    # but strictly speaking, the vector field is now well-defined.
    
    # Factor to balance loss magnitude (optional, kept from stability patch for now)
    # If sigma_min=1e-4, v can be 1e4. Muting it by dt makes it O(1).
    dt = torch.clamp(1.0 - t, min=sigma_min).unsqueeze(-1)
    factor = dt # Use dt as weighting factor to keep loss scale manageable ( ~ Integral v*dt )
    
    # Clip extreme velocity values
    clip_val_trans = v_clip
    clip_val_rot = v_clip
    clip_val_ang = v_clip

    losses = []
    if mask.any():
        # apply factor scaling and clamp
        vp0 = (v_pred[0] * factor).clamp(min=-clip_val_trans, max=clip_val_trans)
        vt0 = (v_true[0] * factor).clamp(min=-clip_val_trans, max=clip_val_trans)

        vp1 = (v_pred[1] * factor).clamp(min=-clip_val_rot, max=clip_val_rot)
        vt1 = (v_true[1] * factor).clamp(min=-clip_val_rot, max=clip_val_rot)

        vp2 = (v_pred[2] * factor).clamp(min=-clip_val_ang, max=clip_val_ang)
        vt2 = (v_true[2] * factor).clamp(min=-clip_val_ang, max=clip_val_ang)

        # Use Huber / SmoothL1
        losses.append(w_trans * F.smooth_l1_loss(vp0[mask], vt0[mask], reduction='mean', beta=huber_beta))
        losses.append(w_rot * F.smooth_l1_loss(vp1[mask], vt1[mask], reduction='mean', beta=huber_beta))
        losses.append(w_ang * F.smooth_l1_loss(vp2[mask], vt2[mask], reduction='mean', beta=huber_beta))
    else:
        losses.append(trans_t.new_tensor(0.0))

    return sum(losses)


def symmetry_consistency_loss(pred_out: Tuple[torch.Tensor, ...], pred_sym_out: Tuple[torch.Tensor, ...], 
                              x_t: Tuple[torch.Tensor, ...], x_t_sym: Tuple[torch.Tensor, ...],
                              t: torch.Tensor, mask: torch.Tensor, sigma_min: float = 1e-4,
                              v_clip: float = 50.0, huber_beta: float = 0.25) -> torch.Tensor:
    """Enforce Parity(v(z)) == v(Parity(z)).
    
    Computes velocity vectors from predicted states and checks equivariance.
    Eq 3: L_sym = || v(z,t) - P^-1 v(P(z), t) ||^2
    """
    
    # 1. Compute velocity v(z)
    v_pred = predicted_velocity(pred_out[0], pred_out[1], pred_out[2], 
                                x_t[0], x_t[1], x_t[2], t, sigma_min=sigma_min)
                                
    # 2. Compute velocity v(P(z))
    v_pred_sym = predicted_velocity(pred_sym_out[0], pred_sym_out[1], pred_sym_out[2],
                                    x_t_sym[0], x_t_sym[1], x_t_sym[2], t, sigma_min=sigma_min)
    
    # 3. Apply Parity Inverse to v_pred_sym
    target_v0 = parity_positions(v_pred_sym[0])
    target_v1 = parity_rotvec(v_pred_sym[1]) 
    target_v2 = parity_torsions(v_pred_sym[2])
    
    # Stabilization: Scale and Clip to prevent numerical explosion near t=1
    # Matches the stability factor used in flow_matching_loss logic
    denom = 1.0 - (1.0 - sigma_min) * t
    factor = torch.clamp(denom, min=sigma_min).unsqueeze(-1)
    
    clip_val = v_clip
    
    losses = []
    if mask.any():
        # Translation
        vp0 = (v_pred[0] * factor).clamp(min=-clip_val, max=clip_val)
        vt0 = (target_v0 * factor).clamp(min=-clip_val, max=clip_val)
        losses.append(F.smooth_l1_loss(vp0[mask], vt0[mask], reduction='mean', beta=huber_beta))
        
        # Rotation
        vp1 = (v_pred[1] * factor).clamp(min=-clip_val, max=clip_val)
        vt1 = (target_v1 * factor).clamp(min=-clip_val, max=clip_val)
        losses.append(F.smooth_l1_loss(vp1[mask], vt1[mask], reduction='mean', beta=huber_beta))
        
        # Angle
        vp2 = (v_pred[2] * factor).clamp(min=-clip_val, max=clip_val)
        vt2 = (target_v2 * factor).clamp(min=-clip_val, max=clip_val)
        losses.append(F.smooth_l1_loss(vp2[mask], vt2[mask], reduction='mean', beta=huber_beta))

    else:
        losses.append(v_pred[0].new_tensor(0.0))
        
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
                             pred_out: Optional[Dict[str, torch.Tensor]] = None, v_pred: Optional[Tuple[torch.Tensor, ...]] = None, sigma_min: float = 1e-4) -> torch.Tensor:
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

    # Pass states to compute velocities internally
    return symmetry_consistency_loss(pred_states, pred_sym_states, 
                                     (trans_t, rot_t, ang_t), (trans_p, rot_p, ang_p),
                                     t, mask, sigma_min=sigma_min)


def riemannian_distance(trans_a: torch.Tensor, rot_a: torch.Tensor, ang_a: torch.Tensor, trans_b: torch.Tensor, rot_b: torch.Tensor, ang_b: torch.Tensor) -> torch.Tensor:
	d_trans = safe_norm(trans_a - trans_b, dim=-1)
	d_rot = so3.geodesic_distance(rot_a, rot_b)
	d_ang = safe_norm(torus.tor_logmap(ang_a, ang_b), dim=-1)
	return torch.sqrt(d_trans.square() + d_rot.square() + d_ang.square())
