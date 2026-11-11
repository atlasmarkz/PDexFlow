"""Model-level sanity checks (CPU, tiny model): forward/backward, sampling validity and decoder handedness."""
import math

import torch

from src.analysis.utils import chirality_sign
from src.data import so3
from src.data.pep_constants import AA, BBHeavyAtom
from src.models.chiralformer import PDexFlowModel


def _batch(B=2, L=8, n_pep=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    ca = torch.arange(L, dtype=torch.float32)[None, :, None] * torch.tensor([3.8, 0.0, 0.0]) + 0.3 * torch.randn(B, L, 3, generator=g)
    pos = torch.zeros(B, L, 15, 3)
    pos[:, :, BBHeavyAtom.N] = ca + torch.tensor([-1.0, 1.2, 0.2])
    pos[:, :, BBHeavyAtom.CA] = ca
    pos[:, :, BBHeavyAtom.C] = ca + torch.tensor([0.8, -0.9, 1.1])
    gen = torch.zeros(B, L, dtype=torch.bool)
    gen[:, :n_pep] = True
    return {
        "aa": torch.randint(0, 20, (B, L), generator=g),
        "pos_heavyatom": pos,
        "torsion_angle": torch.rand(B, L, 5, generator=g) * 2 * math.pi,
        "res_mask": torch.ones(B, L, dtype=torch.bool),
        "generate_mask": gen,
        "res_nb": torch.arange(L)[None].repeat(B, 1),
        "chirality": torch.tensor([0, 1][:B]),
    }


def _model(**kw):
    torch.manual_seed(0)
    return PDexFlowModel(d_model=32, n_layers=2, n_heads=4, dropout=0.0, max_len=64, esm=None, use_ot=False, **kw)


def test_forward_backward_finite():
    model = _model()
    out = model(_batch())
    for key in ("loss", "cfm_loss", "sym_loss", "seq_loss"):
        assert key in out and torch.isfinite(out[key]), key
    assert out["sym_loss"] > 0  # sym_weight defaults to 1, so the second (mirrored) pass runs
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)


def test_sampling_produces_valid_states():
    model = _model().eval()
    b = _batch()
    out = model.sample(b, num_steps=6)
    assert out["trans"].shape == (2, 8, 3) and out["rotmats"].shape == (2, 8, 3, 3) and out["angles"].shape == (2, 8, 5)
    assert torch.isfinite(out["trans"]).all()
    R = out["rotmats"]
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R.transpose(-1, -2) @ R, eye, atol=1e-4), "sampled frames are not orthonormal"
    assert torch.allclose(torch.linalg.det(R), torch.ones(R.shape[:-2]), atol=1e-4), "sampled frames are not proper rotations"
    assert (out["angles"] >= 0).all() and (out["angles"] < 2 * math.pi + 1e-5).all()
    assert ((out["seq"] >= 0) & (out["seq"] < 21)).all()


def test_decoder_handedness_follows_chirality_token():
    g = torch.Generator().manual_seed(1)
    B, L = 2, 6
    aa = torch.randint(0, 20, (B, L), generator=g)
    aa[aa == int(AA.GLY)] = 0  # glycine has no stereocentre
    q = torch.randn(B, L, 4, generator=g)
    rot = so3.quat_to_rotmat(q / q.norm(dim=-1, keepdim=True))
    trans = torch.randn(B, L, 3, generator=g) * 4
    ang = torch.rand(B, L, 5, generator=g) * 2 * math.pi
    pep = torch.ones(B, L, dtype=torch.bool)
    model = _model()
    for token, expected_positive in ((0, True), (1, False)):
        pos15, mask15 = model.decode_to_heavyatom15(aa, trans, rot, ang, torch.full((B,), token), pep, hard_constraints=True)
        P = lambda a: pos15[:, :, a]  # noqa: E731
        s = chirality_sign(P(BBHeavyAtom.N), P(BBHeavyAtom.CA), P(BBHeavyAtom.C), P(BBHeavyAtom.CB))
        assert mask15[..., BBHeavyAtom.CB].all()
        assert ((s > 0) if expected_positive else (s < 0)).all(), f"token {token}: wrong handedness"
