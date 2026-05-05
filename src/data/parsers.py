import os
from typing import Dict, Optional, Tuple

import torch
from Bio.PDB import PDBParser, Selection
from easydict import EasyDict

from src.data.pep_constants import AA, max_num_heavyatoms, restype_to_heavyatom_names, BBHeavyAtom


def _get_residue_heavyatom_info(res) -> Tuple[torch.Tensor, torch.Tensor]:
    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float32)
    mask_heavyatom = torch.zeros([max_num_heavyatoms], dtype=torch.bool)
    restype = AA(res.get_resname())
    for idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
        if atom_name == "":
            continue
        if atom_name in res:
            pos_heavyatom[idx] = torch.tensor(res[atom_name].get_coord(), dtype=torch.float32)
            mask_heavyatom[idx] = True
    return pos_heavyatom, mask_heavyatom


def parse_pdb(path: str, model_id: int = 0, unknown_threshold: float = 1.0, x_mirror: bool = False) -> Tuple[Optional[Dict], Optional[Dict]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(None, path)
    chains = Selection.unfold_entities(structure[model_id], 'C')
    chains.sort(key=lambda c: c.get_id())

    data = EasyDict({
        'chain_id': [],
        'chain_nb': [],
        'resseq': [],
        'icode': [],
        'res_nb': [],
        'aa': [],
        'pos_heavyatom': [],
        'mask_heavyatom': [],
    })

    count_aa, count_unk = 0, 0

    for i, chain in enumerate(chains):
        seq_this = 0
        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))
        for res in residues:
            resname = res.get_resname()
            if not AA.is_aa(resname):
                continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')):
                continue

            restype = AA(resname)
            count_aa += 1
            if restype == AA.UNK:
                count_unk += 1
                continue

            data.chain_id.append(chain.get_id())
            data.chain_nb.append(i)
            data.aa.append(int(restype))

            pos_heavyatom, mask_heavyatom = _get_residue_heavyatom_info(res)
            if x_mirror:
                pos_heavyatom[..., 0] *= -1
            data.pos_heavyatom.append(pos_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)

            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]
            if seq_this == 0:
                seq_this = 1
            else:
                d_ca = torch.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                if d_ca <= 4.0:
                    seq_this += 1
                else:
                    d_resseq = resseq_this - data.resseq[-1]
                    seq_this += max(2, d_resseq)

            data.resseq.append(resseq_this)
            data.icode.append(icode_this)
            data.res_nb.append(seq_this)

    if len(data.aa) == 0:
        return None, None
    if count_aa > 0 and (count_unk / count_aa) >= unknown_threshold:
        return None, None

    # stack tensors
    data.chain_nb = torch.tensor(data.chain_nb, dtype=torch.long)
    data.resseq = torch.tensor(data.resseq, dtype=torch.long)
    data.res_nb = torch.tensor(data.res_nb, dtype=torch.long)
    data.aa = torch.tensor(data.aa, dtype=torch.long)
    data.pos_heavyatom = torch.stack(data.pos_heavyatom, dim=0)
    data.mask_heavyatom = torch.stack(data.mask_heavyatom, dim=0)

    seq_map = {(ch, int(rs), ic): idx for idx, (ch, rs, ic) in enumerate(zip(data.chain_id, data.resseq.tolist(), data.icode))}
    return data, seq_map


def pepmerge_paths(sample_dir: str) -> Dict[str, str]:
    return {
        'peptide': os.path.join(sample_dir, 'peptide.pdb'),
        'pocket': os.path.join(sample_dir, 'pocket.pdb'),
        'receptor': os.path.join(sample_dir, 'receptor.pdb'),
        'pocket_merge': os.path.join(sample_dir, 'pocket_merge.pdb'),
        'receptor_merge': os.path.join(sample_dir, 'receptor_merge.pdb'),
    }
