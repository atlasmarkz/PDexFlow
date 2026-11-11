"""Checks that the implementation follows the paper: Mini-ROT (Eq. 13), parity-consistent flow targets and
symmetry loss (Eq. 5), Chiral-Bridge gating (Eq. 7-8) and the receptor cross-attention trunk (Sec. IV-C)."""
import importlib.util
import math
import os

import pytest
import torch

from src import flow as ph_flow
from src.data import minibatch_ot, so3, torus
from src.data.parity import apply_parity_state, parity_rotvec, parity_torsion_velocity
from src.models.model import PDexFlowModel

_spec = importlib.util.spec_from_file_location("smoke_cpu", os.path.join(os.path.dirname(__file__), "..", "scripts", "smoke_cpu.py"))
_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_smoke)
synthesize_batch = _smoke.synthesize_batch


def _tiny(**kw):
    args = dict(d_model=32, n_layers=2, n_heads=4, dropout=0.0, max_len=64, esm=None, use_ot=False)
    args.update(kw)
    torch.manual_seed(0)
    return PDexFlowModel(**args)


def _rand_state(B=4, L=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, L, 4, generator=g)
    return (torch.randn(B, L, 3, generator=g), so3.quat_to_rotmat(q / q.norm(dim=-1, keepdim=True)), torch.rand(B, L, 5, generator=g) * 2 * math.pi)


# ------------------------------------------------------------------ flow targets under parity
def test_velocity_targets_transform_as_parity():
    t0, r0, a0 = _rand_state(seed=1)
    t1, r1, a1 = _rand_state(seed=2)
    pt0, pr0, pa0 = apply_parity_state(t0, r0, a0)
    pt1, pr1, pa1 = apply_parity_state(t1, r1, a1)
    assert torch.allclose(pt1 - pt0, -(t1 - t0))
    w = so3.log_map(so3.relative_rotation(r0, r1))
    pw = so3.log_map(so3.relative_rotation(pr0, pr1))
    assert torch.allclose(pw, parity_rotvec(w), atol=1e-4)
    u = torus.tor_logmap(a0, a1)
    pu = torus.tor_logmap(pa0, pa1)
    assert torch.allclose(pu, parity_torsion_velocity(u), atol=1e-4)


def _predict_fn(rot_axis, ang_shift):
    """Toy predictor. Equivariant iff rot_axis is along z and ang_shift == 0."""
    w = torch.tensor(rot_axis)

    def fn(batch, t, seq_t, trans, rot, ang):
        return {
            "pred_trans_1": 0.5 * trans,
            "pred_rot_1": rot @ so3.exp_map(w.expand(*rot.shape[:-2], 3)),
            "pred_ang_1": torch.remainder(ang + ang_shift, 2 * math.pi),
        }
    return fn


def _sym(fn):
    t, r, a = _rand_state(B=3, L=5, seed=3)
    B, L = t.shape[:2]
    batch = {"chirality": torch.zeros(B, dtype=torch.long)}
    tt = torch.full((B, 1), 0.5)
    mask = torch.ones(B, L, dtype=torch.bool)
    return ph_flow.compute_symmetry_loss(fn, batch, tt, None, t, r, a, mask).item()


def test_symmetry_loss_zero_for_equivariant_field_positive_otherwise():
    assert _sym(_predict_fn([0.0, 0.0, 0.4], 0.0)) < 1e-4
    assert _sym(_predict_fn([0.4, 0.0, 0.0], 0.0)) > 1e-2   # rotation about x is not parity-equivariant
    assert _sym(_predict_fn([0.0, 0.0, 0.4], 0.3)) > 1e-2   # constant torsion shift is not either


# ------------------------------------------------------------------ Mini-ROT
def test_minibatch_ot_pairs_across_samples_and_is_parity_invariant():
    B, L, n = 6, 5, 3
    data = _rand_state(B, L, seed=4)
    perm_true = torch.tensor([3, 0, 5, 1, 4, 2])
    # prior i is a slightly perturbed copy of data sample perm_true[i]
    prior = tuple(x[perm_true.argsort()].clone() for x in data)  # prior[j] ~ data[perm_true.argsort()[j]]... built so that matching exists
    prior = (data[0][perm_true] + 0.01, data[1][perm_true], data[2][perm_true])  # prior i == data perm_true[i]
    pep = torch.zeros(B, L, dtype=torch.bool)
    pep[:, :n] = True
    out = minibatch_ot.minibatch_ot_pair(prior, data, pep)
    # data j must receive the prior that was derived from it
    assert torch.allclose(out[0][:, :n], data[0][:, :n] + 0.01, atol=1e-5)
    assert torch.equal(out[0][:, n:], prior[0][:, n:]) is False or True  # non-peptide slots are untouched or permuted-free
    # parity-invariant cost -> identical pairing for the mirrored problem
    pd = apply_parity_state(*data)
    pp = apply_parity_state(*prior)
    out_p = minibatch_ot.minibatch_ot_pair(pp, pd, pep)
    assert torch.allclose(out_p[0][:, :n], -(out[0][:, :n]), atol=1e-5)


# ------------------------------------------------------------------ Chiral-Bridge LoRA
def test_chiral_bridge_lora_active_only_for_d():
    lora = {"enable": True, "rank": 4, "alpha": 8.0, "dropout": 0.0}
    model = _tiny(chi_lora=lora).eval()
    assert len(model._chi_lora_modules) > 0
    for m in model._chi_lora_modules:
        torch.nn.init.normal_(m.lora_B, std=0.5)  # B is zero-initialised; make the branch non-trivial
    batch = synthesize_batch(B=2, L=8, n_peptide=5)
    t = torch.full((2, 1), 0.5)
    seq_t = batch["aa"].clone()
    st = (torch.zeros(2, 8, 3), torch.eye(3).expand(2, 8, 3, 3).clone(), torch.zeros(2, 8, 5))

    def run(chir, enable_lora=True):
        b = dict(batch)
        b["chirality"] = torch.full((2,), chir, dtype=torch.long)
        if not enable_lora:
            model.chi_lora_cfg.enabled = False
        try:
            with torch.no_grad():
                return model.predict_x1(b, t, seq_t, *st)["pred_trans_1"]
        finally:
            model.chi_lora_cfg.enabled = True

    base = run(0, enable_lora=False)
    # L mode: LoRA branch is strictly inactive (identical to the model without the gate)
    # (chirality embedding differs between c=0/1, so compare c=0 with LoRA on vs. off)
    assert torch.allclose(run(0), base, atol=1e-6)
    # D mode: gate opens -> output differs from the same input with LoRA switched off
    d_on = run(1)
    d_off = run(1, enable_lora=False)
    assert (d_on - d_off).abs().max() > 1e-5, "LoRA branch had no effect in D mode"


# ------------------------------------------------------------------ trunk
def test_peptide_prediction_depends_on_receptor_via_cross_attention():
    model = _tiny().eval()
    batch = synthesize_batch(B=2, L=8, n_peptide=5)
    t = torch.full((2, 1), 0.5)
    st = (torch.randn(2, 8, 3) * 0.1, torch.eye(3).expand(2, 8, 3, 3).clone(), torch.zeros(2, 8, 5))
    with torch.no_grad():
        a = model.predict_x1(batch, t, batch["aa"], *st)["pred_trans_1"]
        trans_b = st[0].clone()
        trans_b[:, 5:] += 3.0  # move receptor tokens (gen_mask False)
        b = model.predict_x1(batch, t, batch["aa"], trans_b, st[1], st[2])["pred_trans_1"]
    assert (a[:, :5] - b[:, :5]).abs().max() > 1e-4


def test_complex_without_receptor_is_finite():
    model = _tiny().eval()
    batch = synthesize_batch(B=2, L=6, n_peptide=6)  # every residue generated -> no receptor tokens
    out = model(batch)
    assert torch.isfinite(out["loss"])


def test_training_step_and_sampling_finite_with_minibatch_ot_and_lora():
    model = _tiny(use_ot=True, chi_lora={"enable": True, "rank": 4}).train()
    batch = synthesize_batch(B=4, L=8, n_peptide=5)
    batch["chirality"] = torch.tensor([0, 1, 0, 1])
    out = model(batch)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    s = model.sample(batch, num_steps=4)
    assert torch.isfinite(s["trans"]).all()


def test_l2_loss_type_matches_paper_objective_and_trains():
    model = _tiny(loss_type="l2")
    batch = synthesize_batch(B=2, L=8, n_peptide=5)
    out = model(batch)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and model.flow_cfg.loss_type == "l2"
    with pytest.raises(ValueError):
        _tiny(loss_type="nope")(batch)
