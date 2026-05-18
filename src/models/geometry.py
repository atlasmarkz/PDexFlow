import torch


def safe_norm(v: torch.Tensor, dim: int = -1, keepdim: bool = True, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.sum(v * v, dim=dim, keepdim=keepdim) + eps)


def normalize_vector(v: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return v / (safe_norm(v, dim=dim, keepdim=True, eps=eps))


def project_v2v(v: torch.Tensor, e: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Construct an orthonormal basis from 3 points (CA, C, N).

    Returns a matrix with columns [e1,e2,e3] in shape (..., 3, 3).
    For degenerate cases (all points at origin or collinear), returns identity.
    """
    v1 = p1 - center
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)

    # Detect degenerate cases: if center, p1, p2 are all near-zero,
    # the basis is undefined. Replace with identity.
    # Also handles collinear cases where cross product is near-zero.
    is_degenerate = (center.abs() < 1e-6).all(dim=-1) & (p1.abs() < 1e-6).all(dim=-1) & (p2.abs() < 1e-6).all(dim=-1)
    # Also check if e3 is near-zero (collinear atoms)
    e3_norm = torch.linalg.norm(e3, dim=-1, keepdim=True)
    is_collinear = (e3_norm < 1e-6).squeeze(-1)
    is_invalid = is_degenerate | is_collinear

    if is_invalid.any():
        I = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(R)
        is_invalid_expanded = is_invalid.unsqueeze(-1).unsqueeze(-1).expand_as(R)
        R = torch.where(is_invalid_expanded, I, R)

    return R


def compose_rotation_and_translation(R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor):
    R_new = torch.matmul(R1, R2)
    t_new = torch.matmul(R1, t2.unsqueeze(-1)).squeeze(-1) + t1
    return R_new, t_new


def compose_chain(Ts):
    Ts = list(Ts)
    while len(Ts) >= 2:
        R1, t1 = Ts[-2]
        R2, t2 = Ts[-1]
        Ts = Ts[:-2] + [compose_rotation_and_translation(R1, t1, R2, t2)]
    return Ts[0]
