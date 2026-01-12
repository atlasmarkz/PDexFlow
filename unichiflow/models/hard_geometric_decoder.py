"""Hard geometric decoder for L/D templates."""
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from unichiflow.data.pep_constants import BBHeavyAtom
from unichiflow.data.torsion import full_atom_reconstruction, get_heavyatom_mask


class HardGeometricDecoder:
    """Non-parametric decoder that switches L/D frames using a reflection matrix."""

    def __init__(self):
        # We need to un-rotate the frames predicted by the network.
        # Network frame R_pred = R_L @ diag(-1, -1, 1).
        # We want R_dec = R_L (Standard L-frame for reconstruction).
        # So M must be diag(-1, -1, 1).
        self._mirror = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))

    @torch.no_grad()
    def __call__(self, aa: torch.Tensor, trans: torch.Tensor, rotmats: torch.Tensor, angles: torch.Tensor, chirality: torch.Tensor, peptide_mask: torch.Tensor):
        """Decode to heavy-atom coordinates.

        Args:
            aa: (B,L) residue indices (0-19).
            trans: (B,L,3) translations.
            rotmats: (B,L,3,3) backbone frames.
            angles: (B,L,5) torsion angles.
            chirality: (B,) 0=L, 1=D.
            peptide_mask: (B,L) bool mask for generated residues.
        Returns:
            pos15: (B,L,15,3) heavy-atom coords.
            mask15: (B,L,15) validity mask.
        """
        device = rotmats.device
        M = self._mirror.to(device=device, dtype=rotmats.dtype)
        rotmats_dec = rotmats.clone()
        chir = chirality.view(-1, 1, 1, 1).to(rotmats.dtype)
        pep = peptide_mask.view(rotmats.shape[0], rotmats.shape[1], 1, 1).to(rotmats.dtype)
        base = torch.eye(3, device=device, dtype=rotmats.dtype)
        # Apply mirror matrix to backbone frames where chirality=1 (D) and mask=True (peptide)
        # L-frame * M = D-frame (reflected)
        rotmats_dec = rotmats_dec @ (base + chir * pep * (M - base))

        # Hard-constraint correction:
        # Reconstruction uses L-amino acid constants.
        # Mirror(L_structure(theta)) = D_structure(-theta).
        # We want D_structure(theta_pred). So we must input -theta_pred to the L-reconstructor.
        # So where we applied the mirror (chir=1, pep=1), we negate the angles.
        flip_mask = (chir * pep).squeeze(-1).squeeze(-1).unsqueeze(-1) # (B, L, 1)
        angles_dec = angles * (1.0 - 2.0 * flip_mask)

        pos14, _, _ = full_atom_reconstruction(R_bb=rotmats_dec, t_bb=trans, angles=angles_dec, aa=aa)
        pos15 = F.pad(pos14, pad=(0, 0, 0, 15 - 14), value=0.0)
        mask15 = get_heavyatom_mask(aa)
        return pos15, mask15


def reconstruct_atom_positions(frames: Dict[str, torch.Tensor], torsions: torch.Tensor, chirality_token: torch.Tensor, peptide_mask: Optional[torch.Tensor] = None):
    """Reconstruct atom coordinates with hard chirality constraints.

    Args:
        frames: dict containing `aa`, `trans`/`translation`, and `rotmats`/`rotation`.
        torsions: (B,L,5) torsion angles.
        chirality_token: (B,) or (B,1) tensor; 0=L, 1=D.
        peptide_mask: optional (B,L) bool mask; if None defaults to ones.
    Returns:
        pos15: (B,L,15,3) coordinates.
        mask15: (B,L,15) validity mask.
    """
    if not isinstance(frames, dict):
        raise TypeError("frames must be a mapping with aa/trans/rotmats entries")

    aa = frames.get('aa')
    if aa is None:
        aa = frames.get('seq')
    if aa is None:
        raise KeyError("frames must include 'aa' or 'seq'")

    trans = frames.get('trans', frames.get('translation'))
    if trans is None:
        raise KeyError("frames must include 'trans' or 'translation'")

    rotmats = frames.get('rotmats', frames.get('rotation'))
    if rotmats is None:
        raise KeyError("frames must include 'rotmats' or 'rotation'")

    if peptide_mask is None:
        peptide_mask = frames.get('peptide_mask', frames.get('generate_mask'))
    if peptide_mask is None:
        peptide_mask = torch.ones_like(aa, dtype=torch.bool)

    decoder = HardGeometricDecoder()
    return decoder(aa=aa, trans=trans, rotmats=rotmats, angles=torsions, chirality=chirality_token, peptide_mask=peptide_mask)
