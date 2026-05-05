import os
import logging
import pickle
import lmdb
import joblib
import torch
from tqdm.auto import tqdm
from typing import Dict

from src.data.parsers import parse_pdb, pepmerge_paths
from src.data.pep_constants import BBHeavyAtom
from src.data.torsion import get_torsion_angle
from src.data.pep_dataloader import PepDataset


def _center_on_peptide(pep: Dict, rec: Dict):
    ca_mask = pep['mask_heavyatom'][:, BBHeavyAtom.CA]
    center = torch.sum(pep['pos_heavyatom'][ca_mask, BBHeavyAtom.CA], dim=0) / (torch.sum(ca_mask) + 1e-8)
    pep['pos_heavyatom'] -= center[None, None, :]
    rec['pos_heavyatom'] -= center[None, None, :]
    return pep, rec


def _stack_pep_rec(pep: Dict, rec: Dict):
    # concatenate peptide then receptor; generate_mask=1 for peptide, 0 for receptor
    for k in ['aa', 'pos_heavyatom', 'mask_heavyatom', 'chain_id', 'resseq', 'icode', 'res_nb', 'chain_nb']:
        pep[k] = pep[k].clone() if isinstance(pep[k], torch.Tensor) else list(pep[k])
    for k in ['aa', 'pos_heavyatom', 'mask_heavyatom', 'chain_id', 'resseq', 'icode', 'res_nb', 'chain_nb']:
        rec[k] = rec[k].clone() if isinstance(rec[k], torch.Tensor) else list(rec[k])
    out = {}
    out['aa'] = torch.cat([pep['aa'], rec['aa']], dim=0)
    out['pos_heavyatom'] = torch.cat([pep['pos_heavyatom'], rec['pos_heavyatom']], dim=0)
    out['mask_heavyatom'] = torch.cat([pep['mask_heavyatom'], rec['mask_heavyatom']], dim=0)
    out['chain_id'] = pep['chain_id'] + rec['chain_id']
    out['resseq'] = torch.cat([pep['resseq'], rec['resseq']], dim=0)
    out['icode'] = pep['icode'] + rec['icode']
    out['res_nb'] = torch.cat([pep['res_nb'], rec['res_nb']], dim=0)
    out['chain_nb'] = torch.cat([pep['chain_nb'], rec['chain_nb'] + pep['chain_nb'].max() + 1], dim=0)
    gen_mask = torch.zeros_like(out['aa'], dtype=torch.bool)
    gen_mask[: pep['aa'].shape[0]] = True
    return out, gen_mask


def preprocess_pepmerge(task: Dict):
    try:
        paths = pepmerge_paths(task['sample_dir'])
        pep_pdb = paths['peptide']
        rec_pdb = paths['receptor']
        if not os.path.exists(pep_pdb):
            raise FileNotFoundError(f"No peptide.pdb in {task['sample_dir']}")
        if not os.path.exists(rec_pdb):
            raise FileNotFoundError(f"No receptor.pdb in {task['sample_dir']}")

        pep_data, _ = parse_pdb(pep_pdb)
        rec_data, _ = parse_pdb(rec_pdb)
        if pep_data is None or pep_data['aa'].numel() < 3:
            raise ValueError("Peptide has <3 residues or parse failed")
        if rec_data is None or rec_data['aa'].numel() == 0:
            raise ValueError("Receptor parse failed")

        # center coords on peptide
        pep_data, rec_data = _center_on_peptide(pep_data, rec_data)
        pep_data['torsion_angle'], pep_data['torsion_angle_mask'] = get_torsion_angle(pep_data['pos_heavyatom'], pep_data['aa'])

        merged, gen_mask = _stack_pep_rec(pep_data, rec_data)
        merged['generate_mask'] = gen_mask
        merged['torsion_angle'] = torch.cat([pep_data['torsion_angle'], torch.zeros_like(rec_data['aa'], dtype=torch.float32).unsqueeze(-1).repeat(1, pep_data['torsion_angle'].shape[1])], dim=0)
        merged['torsion_angle_mask'] = torch.cat([pep_data['torsion_angle_mask'], torch.zeros_like(rec_data['aa'], dtype=torch.bool).unsqueeze(-1).repeat(1, pep_data['torsion_angle_mask'].shape[1])], dim=0)
        merged['id'] = task['id']
        return merged
    except Exception as e:
        logging.warning('[{}] {}: {}'.format(task['id'], e.__class__.__name__, str(e)))
        return None


class PepMergeDataset(PepDataset):
    """PepMerge conditional dataset: peptide.pdb as generated part, receptor.pdb as fixed condition."""
    def __init__(self, structure_dir, cache_dir, name='pepmerge_cond', reset=False):
        self.structure_dir = structure_dir
        super().__init__(structure_dir=structure_dir, dataset_dir=cache_dir, name=name, x_mirror=False, reset=reset)

    def _load_structures(self, reset):
        if reset:
            if os.path.exists(self._cache_db_path):
                os.remove(self._cache_db_path)
                lock_file = self._cache_db_path + "-lock"
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            self._close_db()

        if not os.path.exists(self._cache_db_path):
            sample_dirs = []
            for root, dirs, files in os.walk(self.structure_dir):
                if 'peptide.pdb' in files and 'receptor.pdb' in files:
                    sample_dirs.append(root)
            if len(sample_dirs) > 0:
                self._preprocess_structures(sample_dirs)

    def _preprocess_structures(self, dir_list):
        tasks = []
        for d in dir_list:
            sample_id = os.path.relpath(d, self.structure_dir).replace(os.sep, '_')
            tasks.append({'id': sample_id, 'sample_dir': d})

        data_list = joblib.Parallel(n_jobs=max(joblib.cpu_count() // 2, 1), )(joblib.delayed(preprocess_pepmerge)(task) for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess'))

        db_conn = lmdb.open(self._cache_db_path, map_size=self.MAP_SIZE, create=True, subdir=False, readonly=False)
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is not None:
                    txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

    def __getitem__(self, index):
        data = super().__getitem__(index)
        data['chirality'] = torch.tensor(1, dtype=torch.long)  # D-peptide
        return data
