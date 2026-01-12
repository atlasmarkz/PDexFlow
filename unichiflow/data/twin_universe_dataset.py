import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from unichiflow.data.parsers import parse_pdb
from unichiflow.data.pep_constants import BBHeavyAtom, PAD_RESIDUE_INDEX, resindex_to_ressymb


# Parity reflection defined in the prompt: P(R, t) = (R · I_flip, -t) with I_flip = diag(1, 1, -1).
# To keep rotations on SO(3) for stable log/exp maps, we use the equivalent SO(3) factor
#   R_parity = R · I_SO3, where I_SO3 = diag(-1, -1, 1).
# This ensures that, when combined with the HardGeometricDecoder's local-frame flip diag(1,1,-1),
# atom positions satisfy x' = -(R @ x + t) while keeping det(R_parity)=+1.
I_FLIP = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
I_SO3 = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))


def apply_parity_transformation(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Apply the P(R,t)=(R·I_flip,-t) transform and mirror coordinates when present.

    The function is batch/shape agnostic and only mutates fields that exist. Expected keys:
      - "rotation" or "rot": (...,3,3)
      - "translation" or "trans": (...,3)
      - "coords": (...,3)
    A shallow copy is returned to avoid altering the input dict in-place.
    """

    mirrored = dict(sample)

    if 'rotation' in mirrored:
        R = mirrored['rotation']
        M = I_SO3.to(device=R.device, dtype=R.dtype)
        mirrored['rotation'] = torch.matmul(R, M)
    if 'rot' in mirrored and 'rotation' not in mirrored:
        R = mirrored['rot']
        M = I_SO3.to(device=R.device, dtype=R.dtype)
        mirrored['rot'] = torch.matmul(R, M)

    if 'translation' in mirrored:
        mirrored['translation'] = -mirrored['translation']
    if 'trans' in mirrored and 'translation' not in mirrored:
        mirrored['trans'] = -mirrored['trans']

    if 'coords' in mirrored:
        mirrored['coords'] = -mirrored['coords']

    if 'pos_heavyatom' in mirrored:
        mirrored['pos_heavyatom'] = -mirrored['pos_heavyatom']
    
    if 'torsion_angle' in mirrored:
        mirrored['torsion_angle'] = -mirrored['torsion_angle']

    return mirrored


class TwinUniverseDataset(Dataset):
    """Minimal PDB dataset with on-the-fly chirality augmentation.

    Each sample returns backbone coordinates, residue tokens, chirality token and mask.
      - 50%: L-mode (native)
      - 50%: D-mode (mirrored on x-axis, token offset by +20)
    """

    def __init__(
        self,
        pdb_paths: Sequence[str],
        mirror_prob: float = 0.5,
        unknown_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.pdb_paths: List[str] = list(pdb_paths)
        self.mirror_prob = float(mirror_prob)
        self.unknown_threshold = float(unknown_threshold)
        if len(self.pdb_paths) == 0:
            raise ValueError("pdb_paths must not be empty")

    def __len__(self) -> int:
        return len(self.pdb_paths)

    def _load_native(self, path: str) -> Dict[str, torch.Tensor]:
        data, _ = parse_pdb(path, unknown_threshold=self.unknown_threshold, x_mirror=False)
        if data is None:
            raise ValueError(f"Failed to parse PDB at {path}")
        coords = data['pos_heavyatom'][:, BBHeavyAtom.CA]  # (L,3)
        seq = data['aa']  # (L,)
        mask = torch.ones_like(seq, dtype=torch.bool)
        return {
            'coords': coords,
            'seq': seq,
            'mask': mask,
            'id': os.path.basename(path),
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.pdb_paths[index]
        sample = self._load_native(path)

        do_mirror = torch.rand(()) < self.mirror_prob
        if do_mirror:
            sample = apply_parity_transformation(sample)
            sample['seq'] = sample['seq'] + 20  # shift to [20,39]
            sample['chirality_token'] = torch.tensor(1, dtype=torch.long)
        else:
            sample['chirality_token'] = torch.tensor(0, dtype=torch.long)

        return sample


class TwinUniverseCollate:
    """Pad variable-length samples from TwinUniverseDataset."""

    def __init__(self, pad_to_multiple_of: Optional[int] = 8, seq_pad_value: int = PAD_RESIDUE_INDEX):
        self.pad_to_multiple_of = pad_to_multiple_of
        self.seq_pad_value = int(seq_pad_value)

    @staticmethod
    def _pad_tensor(x: torch.Tensor, target_len: int, pad_value: float) -> torch.Tensor:
        if x.shape[0] == target_len:
            return x
        pad_shape = (target_len - x.shape[0],) + x.shape[1:]
        pad = torch.full(pad_shape, pad_value, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(int(item['seq'].shape[0]) for item in batch)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            m = int(self.pad_to_multiple_of)
            max_len = int((max_len + m - 1) // m * m)

        padded = []
        for item in batch:
            L = int(item['seq'].shape[0])
            coords = self._pad_tensor(item['coords'], max_len, pad_value=0.0)
            seq = self._pad_tensor(item['seq'], max_len, pad_value=self.seq_pad_value)
            mask = self._pad_tensor(item['mask'], max_len, pad_value=0)
            padded.append({
                'coords': coords,
                'seq': seq,
                'mask': mask.bool(),
                'chirality_token': item['chirality_token'],
                'id': item.get('id', ''),
            })

        return default_collate(padded)


def tokens_to_string(seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> str:
    """Convert integer tokens (0-39, 20 for pad) to an amino-acid string for ESM."""
    if mask is None:
        valid_len = int(seq.numel())
    else:
        valid_len = int(mask.sum().item())
    seq = seq.view(-1)[:valid_len]
    seq_l = (seq % 20).tolist()
    chars = [resindex_to_ressymb[int(t)] for t in seq_l]
    return ''.join(chars)


def tokens_to_strings(batch_seq: torch.Tensor, batch_mask: Optional[torch.Tensor] = None) -> List[str]:
    strings: List[str] = []
    B = batch_seq.shape[0]
    for i in range(B):
        mask_i = batch_mask[i] if batch_mask is not None else None
        strings.append(tokens_to_string(batch_seq[i], mask_i))
    return strings
