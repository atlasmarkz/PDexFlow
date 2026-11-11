import torch

from src.models.chiralformer import PDexFlowModel
from src.data.minibatch_ot import minibatch_optimal_transport
from src.data.pep_constants import BBHeavyAtom


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

    print("Loss components: loss=%.4f cfm=%.4f sym=%.4f seq=%.4f" % (loss, cfm, sym, seq))
    assert torch.isfinite(torch.tensor([loss, cfm, sym, seq])).all(), "Non-finite loss components"

    # Ensure symmetry loss is non-negative and cfm is computed
    assert sym >= 0 and cfm >= 0, "Loss terms should be non-negative"


def test_minibatch_ot():
    torch.manual_seed(0)
    z0 = torch.randn(1, 4, 3)
    z1 = torch.randn(1, 4, 3)
    z0_perm, perm = minibatch_optimal_transport(z0, z1)
    print("OT perm:", perm)
    assert perm.shape == (1, 4)
    # aligned points should correspond to perm indices
    idx = torch.arange(4)[None, :]
    gathered = z0[0, perm[0]]
    assert torch.allclose(gathered, z0_perm[0])


if __name__ == "__main__":
    test_symmetry_and_loss()
    test_minibatch_ot()
    print("Sanity checks passed.")
