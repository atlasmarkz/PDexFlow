"""End-to-end consistency of the parity operator, the data pipeline and the hard decoder.

L structure --(spatial inversion)--> D structure. Frames/torsions measured from the D coordinates by
the *data pipeline* (construct_3d_basis / get_torsion_angle) must (a) equal the parity action on the
L-side frames/torsions and (b) be decoded by HardGeometricDecoder(c=1) back to the inverted coordinates.
"""
import math

import torch

from src.data import so3
from src.data.parity import parity_rotations, parity_torsions
from src.data.pep_constants import BBHeavyAtom
from src.data.torsion import full_atom_reconstruction, get_torsion_angle
from src.models.geometry import construct_3d_basis
from src.models.hard_geometric_decoder import HardGeometricDecoder


def _make_l_structure(L=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    aa = torch.randint(0, 20, (1, L), generator=g)
    q = torch.randn(1, L, 4, generator=g)
    R = so3.quat_to_rotmat(q / q.norm(dim=-1, keepdim=True))
    t = torch.randn(1, L, 3, generator=g) * 5
    ang = torch.rand(1, L, 5, generator=g) * 2 * math.pi
    pos14, _, _ = full_atom_reconstruction(R, t, ang, aa)
    return aa, pos14


def _measure(aa, pos14):
    pos15 = torch.zeros(1, pos14.shape[1], 15, 3)
    pos15[:, :, :14] = pos14
    R = construct_3d_basis(pos15[:, :, BBHeavyAtom.CA], pos15[:, :, BBHeavyAtom.C], pos15[:, :, BBHeavyAtom.N])
    ang, ang_mask = get_torsion_angle(pos14[0], aa[0])
    return R, pos15[:, :, BBHeavyAtom.CA], ang[None], ang_mask[None]


def _circ_err(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b)).abs()


def test_inversion_acts_on_frames_and_torsions_as_parity():
    aa, pos_l = _make_l_structure()
    R_l, _, ang_l, mask = _measure(aa, pos_l)
    R_d, _, ang_d, _ = _measure(aa, -pos_l)  # D = spatial inversion of L
    assert torch.allclose(R_d, parity_rotations(R_l), atol=1e-4), "frame action != R @ diag(-1,-1,1)"
    err = _circ_err(ang_d, parity_torsions(ang_l))[mask]
    assert err.max() < 1e-3, f"torsion action != negation (max err {err.max():.4f})"


def test_hard_decoder_inverts_data_pipeline_in_d_mode():
    aa, pos_l = _make_l_structure(seed=1)
    pos_d = -pos_l
    R_d, ca_d, ang_d, _ = _measure(aa, pos_d)
    dec = HardGeometricDecoder()
    pep = torch.ones(1, aa.shape[1], dtype=torch.bool)
    out, mask15 = dec(aa, ca_d, R_d, ang_d, torch.ones(1, dtype=torch.long), pep)
    m = mask15[..., :14]
    err = (out[..., :14, :] - pos_d).norm(dim=-1)[m]
    assert err.max() < 1e-2, f"D-mode decode mismatch, max atom error {err.max():.4f} A"


def test_hard_decoder_l_mode_identity():
    aa, pos_l = _make_l_structure(seed=2)
    R_l, ca_l, ang_l, _ = _measure(aa, pos_l)
    pep = torch.ones(1, aa.shape[1], dtype=torch.bool)
    out, mask15 = HardGeometricDecoder()(aa, ca_l, R_l, ang_l, torch.zeros(1, dtype=torch.long), pep)
    m = mask15[..., :14]
    err = (out[..., :14, :] - pos_l).norm(dim=-1)[m]
    assert err.max() < 1e-2, f"L-mode decode mismatch, max atom error {err.max():.4f} A"


def test_so3_log_exp_roundtrip_and_distance():
    from src.data import so3
    g = torch.Generator().manual_seed(5)
    w = torch.randn(64, 3, generator=g)
    w = w / w.norm(dim=-1, keepdim=True) * (torch.rand(64, 1, generator=g) * 3.0 + 0.05)  # angles in (0.05, 3.05) < pi
    R = so3.exp_map(w)
    w_back = so3.log_map(R)
    assert torch.allclose(w_back, w, atol=1e-3), "log_map is not the inverse of exp_map"
    assert w_back.norm(dim=-1).max() <= math.pi + 1e-3
    ang = 0.7
    Rz = so3.exp_map(torch.tensor([0.0, 0.0, ang]))
    assert abs(so3.geodesic_distance(torch.eye(3), Rz).item() - ang) < 1e-4
