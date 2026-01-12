import os
import logging
import pickle
import joblib
import lmdb
import torch
from tqdm.auto import tqdm
from unichiflow.data.parsers import parse_pdb
from unichiflow.data.pep_constants import BBHeavyAtom
from unichiflow.data.torsion import get_torsion_angle
from unichiflow.data.pep_dataloader import PepDataset

def preprocess_real_d_structure(task, x_mirror=False):
    try:
        pdb_path = task['pdb_path']
        # Parse the single PDB file
        # We assume the file contains the peptide (and maybe other things we ignore)
        parsed_data = parse_pdb(pdb_path, x_mirror=x_mirror)[0]
        
        if parsed_data is None or parsed_data['aa'].numel() == 0:
            raise ValueError("No chains found in PDB")

        # Heuristic: Find the first chain that looks like a peptide
        # parsed_data contains all residues from all chains flattened.
        # We need to group by chain_nb.
        
        unique_chains = torch.unique(parsed_data['chain_nb'])
        pep_chain_idx = -1
        
        for chain_idx in unique_chains:
            mask = parsed_data['chain_nb'] == chain_idx
            chain_len = mask.sum().item()
            if chain_len >= 3:
                pep_chain_idx = chain_idx
                break
        
        if pep_chain_idx == -1:
            raise ValueError("No valid peptide chain found (len >= 3)")

        # Extract peptide data
        mask = parsed_data['chain_nb'] == pep_chain_idx
        pep = {}
        for k in parsed_data.keys():
            if isinstance(parsed_data[k], torch.Tensor):
                pep[k] = parsed_data[k][mask]
            elif isinstance(parsed_data[k], list):
                pep[k] = [item for item, m in zip(parsed_data[k], mask) if m]
            else:
                pep[k] = parsed_data[k]

        # Process peptide and find center of mass
        center = torch.sum(pep['pos_heavyatom'][pep['mask_heavyatom'][:, BBHeavyAtom.CA], BBHeavyAtom.CA], dim=0) / (torch.sum(pep['mask_heavyatom'][:, BBHeavyAtom.CA]) + 1e-8)
        pep['pos_heavyatom'] -= center[None, None, :]   # translate pep center
        pep['torsion_angle'], pep['torsion_angle_mask'] = get_torsion_angle(pep['pos_heavyatom'], pep['aa'])  # calc angles after translation

        # Create empty receptor data
        # We need to provide empty tensors for receptor to match the expected format
        # or just rely on generate_mask
        
        # Construct data dict
        # generate_mask: 1 for peptide (generated), 0 for receptor (fixed)
        # Since we have no receptor, it's all 1s
        generate_mask = torch.ones_like(pep['aa'], dtype=torch.bool)
        
        data = {
            'id': task['id'],
            'generate_mask': generate_mask,
            'aa': pep['aa'],
            'pos_heavyatom': pep['pos_heavyatom'],
            'mask_heavyatom': pep['mask_heavyatom'],
            'torsion_angle': pep['torsion_angle'],
            'torsion_angle_mask': pep['torsion_angle_mask'],
            'chain_id': pep['chain_id'],
            'res_nb': pep['res_nb'],
            'resseq': pep['resseq'],
            'icode': pep['icode'],
            'chain_nb': torch.tensor([0]), # Dummy chain nb
        }
        
        # Add chirality token: 1 for D-peptide
        # This will be added in __getitem__ usually, but we can add it here or in the dataset wrapper
        # The TwinUniversePepDataset adds it in __getitem__.
        
        return data

    except Exception as e:
        logging.warning('[{}] {}: {}'.format(task['id'], e.__class__.__name__, str(e)))
        return None

class RealDPepDataset(PepDataset):
    def __init__(self, structure_dir, cache_dir, name='real_d_pep', reset=False):
        # We don't use x_mirror, y_mirror, z_mirror for real data usually, 
        # unless we want data augmentation. Assuming no mirroring for now.
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
            # Walk directory to find all .pdb files
            todo_pdbs = []
            for root, dirs, files in os.walk(self.structure_dir):
                for file in files:
                    if file.endswith('.pdb'):
                        # Store relative path or absolute path?
                        # preprocess_real_d_structure expects pdb_path
                        # We can store the relative path from structure_dir
                        rel_path = os.path.relpath(os.path.join(root, file), self.structure_dir)
                        todo_pdbs.append(rel_path)
            
            if len(todo_pdbs) > 0:
                self._preprocess_structures(todo_pdbs)

    def _preprocess_structures(self, pdb_list):
        tasks = []
        for pdb_fname in pdb_list:
            pdb_path = os.path.join(self.structure_dir, pdb_fname)
            # ID is the filename without extension
            pdb_id = os.path.splitext(os.path.basename(pdb_fname))[0]
            tasks.append({'id': pdb_id, 'pdb_path': pdb_path})

        data_list = joblib.Parallel(n_jobs=max(joblib.cpu_count() // 2, 1), )(
            joblib.delayed(preprocess_real_d_structure)(task, self.x_mirror) for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess'))

        db_conn = lmdb.open(self._cache_db_path, map_size=self.MAP_SIZE, create=True, subdir=False, readonly=False, )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is not None:
                    ids.append(data['id'])
                    txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

    def __getitem__(self, index):
        data = super().__getitem__(index)
        # Add chirality token: 1 for D-peptide
        data['chirality'] = torch.tensor(1, dtype=torch.long)
        return data
