"""Shared test helpers."""
import math

import torch

from src.data.pep_constants import BBHeavyAtom


def synthesize_batch(
    B: int = 2,
    L: int = 8,
    n_peptide: int = 5,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Build a mock batch that passes construct_3d_basis(CA, C, N).

    Backbone is a stretched α-helix-like ribbon: CA along +x at 3.8 Å spacing
    (realistic), with fixed N/C offsets that guarantee N, CA, C are non-colinear
    (so construct_3d_basis returns a valid rotation). Not real biology, but
    geometrically well-conditioned — enough to exercise the training path.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    ca = torch.zeros(B, L, 3)
    for i in range(L):
        ca[:, i] = torch.tensor([3.8 * i, 0.0, 0.0])
    # Per-residue jitter so x0/x1 aren't perfectly periodic.
    ca = ca + 0.3 * torch.randn(B, L, 3, generator=g)

    pos = torch.zeros(B, L, 15, 3)
    pos[:, :, BBHeavyAtom.N] = ca + torch.tensor([-1.0, 1.2, 0.2])
    pos[:, :, BBHeavyAtom.CA] = ca
    pos[:, :, BBHeavyAtom.C] = ca + torch.tensor([0.8, -0.9, 1.1])
    pos[:, :, BBHeavyAtom.O] = ca + torch.tensor([1.5, -1.8, 0.5])

    aa = torch.randint(0, 20, (B, L), generator=g)

    res_mask = torch.ones(B, L, dtype=torch.bool)
    gen_mask = torch.zeros(B, L, dtype=torch.bool)
    gen_mask[:, :n_peptide] = True  # first n_peptide residues = generated peptide

    torsion = (torch.rand(B, L, 5, generator=g) * 2 - 1) * math.pi

    res_nb = torch.arange(L, dtype=torch.long)[None].repeat(B, 1)
    chirality = torch.zeros(B, dtype=torch.long)  # 0 = L-peptide batch

    batch = {
        "aa": aa,
        "pos_heavyatom": pos,
        "torsion_angle": torsion,
        "res_mask": res_mask,
        "generate_mask": gen_mask,
        "res_nb": res_nb,
        "chirality": chirality,
    }
    return {k: v.to(device) for k, v in batch.items()}
