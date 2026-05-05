import torch


def chirality_sign(N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor, CB: torch.Tensor) -> torch.Tensor:
    """Return signed volume proxy for CA chirality.

    Positive sign typically corresponds to L for standard coordinate convention.
    Shapes: (...,3)
    """
    v1 = N - CA
    v2 = C - CA
    v3 = CB - CA
    return torch.einsum('...i,...i->...', torch.cross(v1, v2, dim=-1), v3)
