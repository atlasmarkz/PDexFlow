import math
import torch


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions (w,x,y,z). Supports batch (...,3,3)."""
    # Numerically stable branchless conversion
    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]

    t0 = 1 + m00 + m11 + m22
    t1 = 1 + m00 - m11 - m22
    t2 = 1 - m00 + m11 - m22
    t3 = 1 - m00 - m11 + m22

    q0 = torch.stack([t0, R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    q1 = torch.stack([R[..., 2, 1] - R[..., 1, 2], t1, R[..., 0, 1] + R[..., 1, 0], R[..., 0, 2] + R[..., 2, 0]], dim=-1)
    q2 = torch.stack([R[..., 0, 2] - R[..., 2, 0], R[..., 0, 1] + R[..., 1, 0], t2, R[..., 1, 2] + R[..., 2, 1]], dim=-1)
    q3 = torch.stack([R[..., 1, 0] - R[..., 0, 1], R[..., 0, 2] + R[..., 2, 0], R[..., 1, 2] + R[..., 2, 1], t3], dim=-1)

    t = torch.stack([t0, t1, t2, t3], dim=-1)
    idx = torch.argmax(t, dim=-1)

    q = torch.where(idx[..., None] == 0, q0, torch.where(idx[..., None] == 1, q1, torch.where(idx[..., None] == 2, q2, q3)))
    q = q * 0.5 / torch.sqrt(torch.clamp(t.gather(-1, idx[..., None]).squeeze(-1), min=1e-8))[..., None]
    return torch.nn.functional.normalize(q, dim=-1)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (w,x,y,z) -> rotation matrix. q shape (...,4)."""
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)

    m10 = 2 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - wx)

    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = ww - xx - yy + zz

    return torch.stack([
        torch.stack([m00, m01, m02], dim=-1),
        torch.stack([m10, m11, m12], dim=-1),
        torch.stack([m20, m21, m22], dim=-1),
    ], dim=-2)


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Quaternion slerp between q0 and q1. t broadcastable to (...,1)."""
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot).clamp(max=1.0)

    # If very close, lerp
    close = dot > 0.9995
    lerp = torch.nn.functional.normalize((1 - t) * q0 + t * q1, dim=-1)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp(min=1e-8)
    theta = theta_0 * t
    sin_theta = torch.sin(theta)

    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    out = s0 * q0 + s1 * q1
    out = torch.nn.functional.normalize(out, dim=-1)
    return torch.where(close, lerp, out)


def geodesic_t(t: torch.Tensor, R1: torch.Tensor, R0: torch.Tensor) -> torch.Tensor:
    """Interpolate from R0 to R1 at time t in [0,1]."""
    q0 = rotmat_to_quat(R0)
    q1 = rotmat_to_quat(R1)
    qt = slerp(q0, q1, t)
    return quat_to_rotmat(qt)


def _hat(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix from vector (Rodrigues helper)."""
    vx, vy, vz = v.unbind(dim=-1)
    O = torch.zeros_like(vx)
    return torch.stack([
        torch.stack([O, -vz, vy], dim=-1),
        torch.stack([vz, O, -vx], dim=-1),
        torch.stack([-vy, vx, O], dim=-1),
    ], dim=-2)


def _vee(M: torch.Tensor) -> torch.Tensor:
    """Inverse of _hat: skew-symmetric matrix -> vector."""
    return torch.stack([M[..., 2, 1], M[..., 0, 2], M[..., 1, 0]], dim=-1)


def relative_rotation(R_from: torch.Tensor, R_to: torch.Tensor) -> torch.Tensor:
    """Compute relative rotation R_rel such that R_to = R_from @ R_rel."""
    return torch.matmul(R_from.transpose(-1, -2), R_to)


def exp_map(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rodrigues' formula for Exp: axis-angle (omega) -> rotation matrix."""
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp(min=eps)
    k = omega / theta
    K = _hat(k)
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(*omega.shape[:-1], 3, 3)
    # Use series expansion when angle is tiny to avoid numerical issues
    small = (theta < 1e-4).float()[..., None]
    sin_term = sin_t + small * (theta[..., None] - sin_t)
    cos_term = (1 - cos_t) + small * (0.5 * theta.square()[..., None] - (1 - cos_t))
    return I + sin_term * K + cos_term * torch.matmul(K, K)


def log_map(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Logarithm map on SO(3): rotation matrix -> axis-angle vector."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1) * 0.5).clamp(min=-1.0, max=1.0)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta).clamp(min=eps)
    skew = 0.5 * (R - R.transpose(-1, -2))
    omega = _vee(skew)
    scale = theta / sin_theta
    omega = omega * scale[..., None]
    # Use first-order approximation for very small angles
    mask = (theta < 1e-4).float()[..., None]
    omega = omega * (1 - mask) + omega * 0.5 * theta[..., None] * mask
    return omega


def geodesic_distance(R1: torch.Tensor, R2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Riemannian geodesic distance between two rotation matrices."""
    rel = relative_rotation(R1, R2)
    omega = log_map(rel, eps=eps)
    return torch.linalg.norm(omega, dim=-1)
