import warnings

import torch
from Bio import BiopythonWarning
from Bio.PDB import PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder

from unichiflow.data.pep_constants import AA, restype_to_heavyatom_names


def save_pdb(data, path: str):
    def _mask_select(v, mask):
        if isinstance(v, str):
            return ''.join([s for i, s in enumerate(v) if mask[i]])
        if isinstance(v, list):
            return [s for i, s in enumerate(v) if mask[i]]
        if isinstance(v, torch.Tensor):
            return v[mask]
        return v

    def _build_chain(builder, aa_ch, pos_heavyatom_ch, mask_heavyatom_ch, chain_id_ch, resseq_ch, icode_ch):
        builder.init_chain(chain_id_ch[0])
        builder.init_seg('    ')
        for aa_res, pos_res, mask_res, resseq_res, icode_res in zip(aa_ch, pos_heavyatom_ch, mask_heavyatom_ch, resseq_ch, icode_ch):
            restype = AA(int(aa_res))
            builder.init_residue(resname=str(restype), field=' ', resseq=int(resseq_res), icode=icode_res)
            for i, atom_name in enumerate(restype_to_heavyatom_names[restype]):
                if atom_name == '':
                    continue
                if (~mask_res[i]).any():
                    continue
                if len(atom_name) == 1:
                    fullname = f' {atom_name}  '
                elif len(atom_name) == 2:
                    fullname = f' {atom_name} '
                elif len(atom_name) == 3:
                    fullname = f' {atom_name}'
                else:
                    fullname = atom_name
                builder.init_atom(atom_name, pos_res[i].tolist(), 0.0, 1.0, ' ', fullname)

    warnings.simplefilter('ignore', BiopythonWarning)
    builder = StructureBuilder()
    builder.init_structure(0)
    builder.init_model(0)

    for ch_nb in torch.unique(data['chain_nb']).tolist():
        mask = (data['chain_nb'] == ch_nb)
        aa = _mask_select(data['aa'], mask)
        pos = _mask_select(data['pos_heavyatom'], mask)
        mpos = _mask_select(data['mask_heavyatom'], mask)
        chain_id = _mask_select(data['chain_id'], mask)
        resseq = _mask_select(data['resseq'], mask)
        icode = _mask_select(data['icode'], mask)
        _build_chain(builder, aa, pos, mpos, chain_id, resseq, icode)

    io = PDBIO()
    io.set_structure(builder.get_structure())
    io.save(path)
