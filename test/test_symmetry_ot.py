import torch

from src.models.chiralformer import PDexFlowModel
from src.data.minibatch_ot import solve_ot


def make_dummy_batch(B: int = 2, L: int = 8, angle_dim: int = 5):
    torch.manual_seed(42)
    aa = torch.randint(low=0, high=20, size=(B, L))
    # heavy-atom coords: (B,L,15,3); we only need CA/C/N columns for basis
    pos = torch.randn(B, L, 15, 3) * 0.1
    generate_mask = torch.ones(B, L, dtype=torch.bool)  # treat all residues as peptide for simplicity
    res_mask = torch.ones(B, L, dtype=torch.bool)
    torsion_angle = torch.rand(B, L, angle_dim) * 2 * torch.pi
    res_nb = torch.arange(L)[None, :].repeat(B, 1)
    chirality = torch.tensor([0, 1], dtype=torch.long)[:B]

    batch = {
        "aa": aa,
        "pos_heavyatom": pos,
        "generate_mask": generate_mask,
        "res_mask": res_mask,
        "torsion_angle": torsion_angle,
        "res_nb": res_nb,
        "chirality": chirality,
    }
    return batch


def test_symmetry_and_loss():
    B, L = 2, 8
    angle_dim = 5
    batch = make_dummy_batch(B=B, L=L, angle_dim=angle_dim)

    model = PDexFlowModel(
        d_model=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        max_len=64,
        aatype_vocab=21,
        sigma_trans=0.5,
        chi_lora={"enable": False},
        esm={"enable": False},
        cfm_weight=1.0,
        sym_weight=1.0,
        angle_dim=angle_dim,
        use_ot=True,
    )

    with torch.no_grad():
        out = model(batch)
    loss = float(out["loss"])
    cfm = float(out["cfm_loss"])
    sym = float(out["sym_loss"])
    seq = float(out["seq_loss"])

    assert torch.isfinite(torch.tensor([loss, cfm, sym, seq])).all(), "Non-finite loss components"

    # Ensure symmetry loss is non-negative and cfm is computed
    assert sym >= 0 and cfm >= 0, "Loss terms should be non-negative"


def test_solve_ot_returns_optimal_assignment():
    import itertools

    torch.manual_seed(0)
    cost = torch.rand(3, 5, 5)
    perm = solve_ot(cost)
    assert perm.shape == (3, 5)
    for b in range(3):
        assert sorted(perm[b].tolist()) == list(range(5)), "not a permutation"
        got = cost[b, torch.arange(5), perm[b]].sum()
        best = min(cost[b, torch.arange(5), torch.tensor(p)].sum() for p in itertools.permutations(range(5)))
        assert torch.isclose(got, best), "assignment is not optimal"
