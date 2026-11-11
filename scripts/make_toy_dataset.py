"""Generate a tiny synthetic PepMerge-style dataset (peptide.pdb + pocket.pdb per complex).

Only meant for smoke-testing the full pipeline (cache -> training -> sampling -> evaluation) without the
real PepMerge data. Geometry is not biologically meaningful.

    python scripts/make_toy_dataset.py --out data/toy/pepmerge --n_train 8 --n_test 2
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import so3  # noqa: E402
from src.data.torsion import full_atom_reconstruction, get_heavyatom_mask  # noqa: E402
from src.data.writers import save_pdb  # noqa: E402


def _chain(n, g, spacing=3.8, origin=None, spread=0.0):
    aa = torch.randint(0, 20, (1, n), generator=g)
    ca = torch.zeros(1, n, 3)
    for i in range(n):  # helix-like ribbon
        a = 100.0 * math.pi / 180.0 * i
        ca[0, i] = torch.tensor([1.6 * math.cos(a), 1.6 * math.sin(a), 1.5 * i])
    ca = ca * (spacing / 3.8) + spread * torch.randn(1, n, 3, generator=g)
    if origin is not None:
        ca = ca + origin
    q = torch.randn(1, n, 4, generator=g) * 0.15 + torch.tensor([1.0, 0.0, 0.0, 0.0])
    R = so3.quat_to_rotmat(q / q.norm(dim=-1, keepdim=True))
    ang = torch.rand(1, n, 5, generator=g) * 2 * math.pi
    pos14, _, _ = full_atom_reconstruction(R, ca, ang, aa)
    pos15 = torch.zeros(1, n, 15, 3)
    pos15[:, :, :14] = pos14
    mask = get_heavyatom_mask(aa)
    return {
        "aa": aa[0],
        "pos_heavyatom": pos15[0],
        "mask_heavyatom": mask[0].bool(),
        "chain_nb": torch.zeros(n, dtype=torch.long),
        "chain_id": ["A"] * n,
        "resseq": torch.arange(1, n + 1),
        "icode": [" "] * n,
    }


def make_complex(path, seed):
    g = torch.Generator().manual_seed(seed)
    os.makedirs(path, exist_ok=True)
    pep = _chain(int(torch.randint(5, 9, (1,), generator=g)), g)
    pep["chain_id"] = ["B"] * len(pep["aa"])
    pocket = _chain(14, g, origin=torch.tensor([9.0, 0.0, 2.0]), spread=2.0)
    save_pdb(pep, os.path.join(path, "peptide.pdb"))
    save_pdb(pocket, os.path.join(path, "pocket.pdb"))    # used for training
    save_pdb(pocket, os.path.join(path, "receptor.pdb"))  # used by sampling / evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy/pepmerge")
    ap.add_argument("--n_train", type=int, default=8)
    ap.add_argument("--n_test", type=int, default=2)
    args = ap.parse_args()

    names_txt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "names.txt")
    test_names = [l.strip() for l in open(names_txt) if l.strip()][: args.n_test]  # ids reserved for the test split
    for i in range(args.n_train):
        make_complex(os.path.join(args.out, f"toy_train_{i:03d}"), seed=i)
    for j, name in enumerate(test_names):
        make_complex(os.path.join(args.out, name), seed=1000 + j)
    print(f"wrote {args.n_train} train + {len(test_names)} test complexes to {args.out}")


if __name__ == "__main__":
    main()
