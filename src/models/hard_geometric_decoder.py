"""Hard geometric decoder for L/D templates."""
import torch
import torch.nn.functional as F

from src.data.parity import LOCAL_REFLECTION
from src.data.torsion import full_atom_reconstruction, get_heavyatom_mask


class HardGeometricDecoder:
    """Chirality-preserving NeRF-style full-atom reconstruction (paper Sec. IV-D, Eq. 10-11).

        X_atom = R_i @ Q_c @ Y_local + x_i,     Y_local = NeRF(a_i, chi_i)
        Q_c = I (c = 0, L-mode),  diag(1, 1, -1) (c = 1, D-mode)

    Q_c is a *local reflection* (det = -1) applied to the residue-local atom coordinates before
    they are placed by the generated backbone frame, so D-mode residues are mirror images of the
    L template by construction.

    Torsion convention: dihedrals are measured on the structure being described, and a mirror
    image flips every dihedral sign. The model predicts the dihedrals of the *D* structure when
    c = 1, hence the L template is built from -chi and then reflected; this makes the decoder the
    exact inverse of the data pipeline (inversion of coordinates -> frames R.diag(-1,-1,1), chi -> -chi).
    This identity is verified numerically in ``test/test_parity_roundtrip.py``.
    """

    def __init__(self):
        self._reflection = LOCAL_REFLECTION.clone()

    @torch.no_grad()
    def __call__(self, aa: torch.Tensor, trans: torch.Tensor, rotmats: torch.Tensor, angles: torch.Tensor, chirality: torch.Tensor, peptide_mask: torch.Tensor):
        """Decode to heavy-atom coordinates.

        Args:
            aa: (B,L) residue indices (0-19).
            trans: (B,L,3) translations (CA positions).
            rotmats: (B,L,3,3) backbone frames.
            angles: (B,L,5) torsion angles (psi, chi1-4).
            chirality: (B,) 0=L, 1=D.
            peptide_mask: (B,L) bool mask for generated residues.
        Returns:
            pos15: (B,L,15,3) heavy-atom coords.
            mask15: (B,L,15) validity mask.
        """
        device, dtype = rotmats.device, rotmats.dtype
        B, L = aa.shape
        chir = chirality.view(-1).to(device=device, dtype=dtype)
        d_mode = (chir.view(B, 1) * peptide_mask.to(dtype)) > 0.5  # (B, L) residues to mirror

        # NeRF in the residue-local frame (identity frame at the origin), L templates.
        eye = torch.eye(3, device=device, dtype=dtype).expand(B, L, 3, 3)
        zero_t = torch.zeros(B, L, 3, device=device, dtype=dtype)
        angles_l = torch.where(d_mode.unsqueeze(-1), -angles, angles)
        y_local, _, _ = full_atom_reconstruction(R_bb=eye, t_bb=zero_t, angles=angles_l, aa=aa)

        # Q_c: local reflection for D residues only.
        q = self._reflection.to(device=device, dtype=dtype)
        y_reflected = y_local @ q.transpose(-1, -2)
        y_local = torch.where(d_mode[..., None, None], y_reflected, y_local)

        # Place with the generated frame: X = R @ Y + x.
        pos14 = torch.matmul(rotmats[:, :, None], y_local.unsqueeze(-1)).squeeze(-1) + trans[:, :, None]
        pos15 = F.pad(pos14, pad=(0, 0, 0, 15 - 14), value=0.0)
        mask15 = get_heavyatom_mask(aa)
        return pos15, mask15
