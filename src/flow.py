"""Parity-Harmonized Flow utilities and losses.

Implements PH-Flow on SE(3)^N x T^{kN} with parity consistency.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from src.data import so3, torus


PARITY_FLIP = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
PARITY_SO3 = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))


def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
	return torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim, keepdim=keepdim), min=eps))


def parity_positions(x: torch.Tensor) -> torch.Tensor:
	return -x


def parity_rotations(R: torch.Tensor) -> torch.Tensor:
	M = PARITY_SO3.to(device=R.device, dtype=R.dtype)
	return torch.matmul(R, M)


def parity_rotvec(omega: torch.Tensor) -> torch.Tensor:
	M = PARITY_SO3.to(device=omega.device, dtype=omega.dtype)
	return torch.matmul(omega, M)


def parity_torsions(theta: torch.Tensor) -> torch.Tensor:
	return -theta


def apply_parity_state(trans: torch.Tensor, rot: torch.Tensor, ang: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	return parity_positions(trans), parity_rotations(rot), parity_torsions(ang)


def target_velocity(trans_0: torch.Tensor, trans_1: torch.Tensor, rot_0: torch.Tensor, rot_1: torch.Tensor, ang_0: torch.Tensor, ang_1: torch.Tensor, sigma_min: float = 1e-4):
    v_trans = trans_1 - trans_0
    rel_rot = so3.relative_rotation(rot_0, rot_1)
    v_rot = so3.log_map(rel_rot)
    v_ang = torus.tor_logmap(ang_0, ang_1)
    return v_trans, v_rot, v_ang


def predicted_velocity(pred_trans_1: torch.Tensor, pred_rot_1: torch.Tensor, pred_ang_1: torch.Tensor, trans_t: torch.Tensor, rot_t: torch.Tensor, ang_t: torch.Tensor, t: torch.Tensor, sigma_min: float = 1e-4):
    dt = 1.0 - t
    dt = torch.clamp(dt, min=sigma_min)[..., None]
    v_trans = (pred_trans_1 - trans_t) / dt
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
    if weights is None:
        w_trans, w_rot, w_ang = 1.0, 1.0, 1.0
    else:
        w_trans, w_rot, w_ang = float(weights[0]), float(weights[1]), float(weights[2])

    # Compute displacements in x1-space first, THEN divide by dt. The old code clamped
    # velocities after the /(1-t) amplification, which meant fp16 could already have
    # overflowed to inf before clamp got a chance to fire. Clamping the x1-space
    # displacement by (v_clip * dt) is mathematically equivalent (|v| <= v_clip) but
    # keeps intermediates finite under AMP.
    dt = torch.clamp(1.0 - t, min=sigma_min)[..., None]
    cap_trans = v_clip * dt
    cap_rot = v_clip * dt
    cap_ang = v_clip * dt

    delta_true_trans = (trans_1 - trans_0).clamp(min=-cap_trans, max=cap_trans)
    delta_pred_trans = (pred_x1[0] - trans_t).clamp(min=-cap_trans, max=cap_trans)

    rel_rot_true = so3.relative_rotation(rot_0, rot_1)
    delta_true_rot = so3.log_map(rel_rot_true).clamp(min=-cap_rot, max=cap_rot)
    rel_rot_pred = so3.relative_rotation(rot_t, pred_x1[1])
    delta_pred_rot = so3.log_map(rel_rot_pred).clamp(min=-cap_rot, max=cap_rot)

    delta_true_ang = torus.tor_logmap(ang_0, ang_1).clamp(min=-cap_ang, max=cap_ang)
    delta_pred_ang = torus.tor_logmap(ang_t, pred_x1[2]).clamp(min=-cap_ang, max=cap_ang)

    losses = []
    if mask.any():
        # Divide by dt once, AFTER clamping, so each velocity is guaranteed in [-v_clip, v_clip].
        vt0 = delta_true_trans / dt
        vp0 = delta_pred_trans / dt
        vt1 = delta_true_rot / dt
        vp1 = delta_pred_rot / dt
        vt2 = delta_true_ang / dt
        vp2 = delta_pred_ang / dt

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
    """Parity-consistency loss: v(P x_t) should equal P v(x_t).

    Velocities are compared in x1-space (clamp-before-divide) for the same AMP-safety
    reason as flow_matching_loss; the parity transform P commutes with per-channel
    clamping because it is a sign flip on each coordinate.
    """
    dt = torch.clamp(1.0 - t, min=sigma_min)[..., None]
    cap = v_clip * dt

    # Main branch: displacements from x_t to pred_x1, clipped in x1-space.
    delta_trans = (pred_out[0] - x_t[0]).clamp(min=-cap, max=cap)
    rel_rot = so3.relative_rotation(x_t[1], pred_out[1])
    delta_rot = so3.log_map(rel_rot).clamp(min=-cap, max=cap)
    delta_ang = torus.tor_logmap(x_t[2], pred_out[2]).clamp(min=-cap, max=cap)

    # Symmetric branch: same clamp applied in parity-mirrored space.
    delta_trans_sym = (pred_sym_out[0] - x_t_sym[0]).clamp(min=-cap, max=cap)
    rel_rot_sym = so3.relative_rotation(x_t_sym[1], pred_sym_out[1])
    delta_rot_sym = so3.log_map(rel_rot_sym).clamp(min=-cap, max=cap)
    delta_ang_sym = torus.tor_logmap(x_t_sym[2], pred_sym_out[2]).clamp(min=-cap, max=cap)

    # Target for main velocity = P applied to sym-branch velocity.
    target_v0 = parity_positions(delta_trans_sym / dt)
    target_v1 = parity_rotvec(delta_rot_sym / dt)
    target_v2 = parity_torsions(delta_ang_sym / dt)

    v_main_0 = delta_trans / dt
    v_main_1 = delta_rot / dt
    v_main_2 = delta_ang / dt

    losses = []
    if mask.any():
        losses.append(F.smooth_l1_loss(v_main_0[mask], target_v0[mask], reduction='mean', beta=huber_beta))
        losses.append(F.smooth_l1_loss(v_main_1[mask], target_v1[mask], reduction='mean', beta=huber_beta))
        losses.append(F.smooth_l1_loss(v_main_2[mask], target_v2[mask], reduction='mean', beta=huber_beta))
    else:
        losses.append(v_main_0.new_tensor(0.0))

    return sum(losses)


def _clone_batch_with_chirality(batch: Dict[str, torch.Tensor], chirality: torch.Tensor) -> Dict[str, torch.Tensor]:
	# Shallow copy: the sym branch only replaces `chirality`. Every other tensor
	# (including cached `esm_cond`) is read-only inside predict_x1, so reusing the
	# same object is safe and saves a full-batch clone per step.
	batch_sym = dict(batch)
	batch_sym['chirality'] = chirality
	return batch_sym


def compute_symmetry_loss(predict_fn, batch: Dict[str, torch.Tensor], t: torch.Tensor, seq_t: torch.Tensor, trans_t: torch.Tensor, rot_t: torch.Tensor, ang_t: torch.Tensor, mask: torch.Tensor,
                             pred_out: Optional[Dict[str, torch.Tensor]] = None, v_pred: Optional[Tuple[torch.Tensor, ...]] = None, sigma_min: float = 1e-4) -> torch.Tensor:
    if pred_out is None:
        pred_out = predict_fn(batch, t, seq_t, trans_t, rot_t, ang_t)

    pred_states = (pred_out['pred_trans_1'], pred_out['pred_rot_1'], pred_out['pred_ang_1'])

    chir = batch.get('chirality', torch.zeros((trans_t.size(0),), device=trans_t.device, dtype=torch.long))
    if isinstance(chir, torch.Tensor) and chir.ndim == 2 and chir.shape[-1] == 1:
        chir = chir.squeeze(-1)
    chir_mirror = torch.ones_like(chir) - chir

    batch_sym = _clone_batch_with_chirality(batch, chir_mirror)
    trans_p, rot_p, ang_p = apply_parity_state(trans_t, rot_t, ang_t)
    out_sym = predict_fn(batch_sym, t, seq_t, trans_p, rot_p, ang_p)

    pred_sym_states = (out_sym['pred_trans_1'], out_sym['pred_rot_1'], out_sym['pred_ang_1'])

    return symmetry_consistency_loss(pred_states, pred_sym_states,
                                     (trans_t, rot_t, ang_t), (trans_p, rot_p, ang_p),
                                     t, mask, sigma_min=sigma_min)


def riemannian_distance(trans_a: torch.Tensor, rot_a: torch.Tensor, ang_a: torch.Tensor, trans_b: torch.Tensor, rot_b: torch.Tensor, ang_b: torch.Tensor) -> torch.Tensor:
	d_trans = safe_norm(trans_a - trans_b, dim=-1)
	d_rot = so3.geodesic_distance(rot_a, rot_b)
	d_ang = safe_norm(torus.tor_logmap(ang_a, ang_b), dim=-1)
	return torch.sqrt(d_trans.square() + d_rot.square() + d_ang.square())