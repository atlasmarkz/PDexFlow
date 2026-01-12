import argparse
from typing import Optional

import torch

from unichiflow.analysis.utils import chirality_sign
from unichiflow.data.parsers import parse_pdb
from unichiflow.data.pep_constants import BBHeavyAtom, AA


def _summarize_chirality(pdb_path: str) -> dict:
    data, _ = parse_pdb(pdb_path, x_mirror=False)
    if data is None:
        raise ValueError(f"Failed to parse PDB: {pdb_path}")

    pos = data.pos_heavyatom  # (L,15,3)
    mask = data.mask_heavyatom  # (L,15)
    aa = data.aa  # (L,)

    # Only residues with CB (skip GLY and missing atoms)
    has_cb = (aa != int(AA.GLY)) & mask[:, BBHeavyAtom.CB] & mask[:, BBHeavyAtom.N] & mask[:, BBHeavyAtom.CA] & mask[:, BBHeavyAtom.C]
    if has_cb.any():
        s = chirality_sign(
            pos[has_cb, BBHeavyAtom.N],
            pos[has_cb, BBHeavyAtom.CA],
            pos[has_cb, BBHeavyAtom.C],
            pos[has_cb, BBHeavyAtom.CB],
        )
        pos_count = int((s > 0).sum().item())
        neg_count = int((s < 0).sum().item())
        zero_count = int((s == 0).sum().item())
        mean = float(s.mean().item())
    else:
        pos_count = neg_count = zero_count = 0
        mean = 0.0

    return {
        'pdb': pdb_path,
        'n_res': int(aa.shape[0]),
        'n_used': int(has_cb.sum().item()),
        'sign_pos': pos_count,
        'sign_neg': neg_count,
        'sign_zero': zero_count,
        'mean_sign': mean,
    }


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Check CA chirality sign from a PDB (N,CA,C,CB signed volume).")
    ap.add_argument('pdb', nargs='+', help='Path(s) to PDB file(s).')
    args = ap.parse_args(argv)

    for p in args.pdb:
        out = _summarize_chirality(p)
        print(
            f"{out['pdb']}: used {out['n_used']}/{out['n_res']} residues | "
            f"sign + {out['sign_pos']}  - {out['sign_neg']}  0 {out['sign_zero']} | mean {out['mean_sign']:.4e}"
        )


if __name__ == '__main__':
    main()
