"""
Minimal forward pass test for PDexFlow on RTX 3080.
No checkpoint needed - tests that the model architecture runs.
"""
import torch

from src.models.chiralformer import PDexFlowModel
from src.data import so3, torus


def make_dummy_batch(B: int = 1, L: int = 8, angle_dim: int = 5, device: str = "cuda"):
    torch.manual_seed(42)
    aa = torch.randint(low=0, high=20, size=(B, L), device=device)
    pos = torch.randn(B, L, 15, 3, device=device) * 2.0
    generate_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    res_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    torsion_angle = torch.rand(B, L, angle_dim, device=device) * 2 * torch.pi
    chirality = torch.zeros(B, device=device, dtype=torch.long)  # (B,) - 0=L, 1=D

    return {
        "aa": aa,
        "pos_heavyatom": pos,
        "generate_mask": generate_mask,
        "res_mask": res_mask,
        "torsion_angle": torsion_angle,
        "chirality": chirality,
    }


def test_forward_pass():
    print("=" * 60)
    print("Test 1: Forward Pass")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = PDexFlowModel(
        d_model=128,
        n_layers=4,
        n_heads=4,
        dropout=0.0,
        max_len=512,
        aatype_vocab=21,
        angle_dim=5,
        sigma_trans=0.5,
        use_ot=False,
        sym_weight=0.0,
        cfm_weight=1.0,
        seq_weight=0.0,
        cfm_w_trans=1.0,
        cfm_w_rot=1.0,
        cfm_w_ang=1.0,
        esm=None,
    ).to(device)

    batch = make_dummy_batch(B=1, L=8, device=device)
    print(f"Batch keys: {list(batch.keys())}")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}")

    with torch.no_grad():
        out = model(batch)

    print(f"\nOutput keys: {list(out.keys())}")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}, mean={v.mean().item():.4f}")
        else:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n[PASS] Forward pass successful")
    return model, batch, out


def test_ode_sampling():
    print("\n" + "=" * 60)
    print("Test 2: ODE Sampling (Euler integration)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PDexFlowModel(
        d_model=128,
        n_layers=4,
        n_heads=4,
        dropout=0.0,
        max_len=512,
        aatype_vocab=21,
        angle_dim=5,
        sigma_trans=0.5,
        use_ot=False,
        sym_weight=0.0,
        cfm_weight=1.0,
        seq_weight=0.0,
        esm=None,
    ).to(device)

    model.eval()
    B, L = 1, 8
    batch = make_dummy_batch(B=B, L=L, device=device)

    trans = torch.randn(B, L, 3, device=device) * 2.0
    rot = so3.quat_to_rotmat(torch.randn(B, L, 4, device=device))
    ang = torch.rand(B, L, 5, device=device) * 2 * torch.pi

    num_steps = 20
    print(f"Sampling with {num_steps} Euler steps...")

    with torch.no_grad():
        for k in range(num_steps):
            t = torch.full((B, 1), (k + 1) / num_steps, device=device)
            pred = model.predict_x1(batch, t, batch["aa"], trans, rot, ang)

            dt = 1.0 / num_steps

            trans = (1 - dt) * trans + dt * pred["pred_trans_1"]
            rot = so3.geodesic_t(
                torch.full((B, L, 1), dt, device=device),
                pred["pred_rot_1"], rot
            )
            ang = torus.tor_geodesic_t(
                torch.full((B, L, 1), dt, device=device),
                pred["pred_ang_1"], ang
            )

            if (k + 1) % 5 == 0:
                print(f"  Step {k+1}/{num_steps}: trans norm={trans.norm():.3f}")

    print(f"\nFinal trans shape: {trans.shape}")
    print(f"Final rot shape: {rot.shape}")
    print(f"Final ang shape: {ang.shape}")
    print("[PASS] ODE sampling successful")
    return trans, rot, ang


def test_memory_usage():
    print("\n" + "=" * 60)
    print("Test 3: Memory Usage on RTX 3080")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("No CUDA available, skipping memory test")
        return

    device = "cuda"
    torch.cuda.reset_peak_memory_stats()

    for B, L in [(1, 8), (2, 16)]:
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1024**2

        model = PDexFlowModel(
            d_model=256,
            n_layers=6,
            n_heads=8,
            dropout=0.0,
            max_len=512,
            aatype_vocab=21,
            angle_dim=5,
            sigma_trans=0.5,
            use_ot=False,
            sym_weight=0.0,
            esm=None,
        ).to(device)

        batch = make_dummy_batch(B=B, L=L, device=device)

        with torch.no_grad():
            _ = model(batch)

        mem_after = torch.cuda.memory_allocated() / 1024**2
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2

        print(f"  B={B}, L={L}: current={mem_after:.1f}MB, peak={mem_peak:.1f}MB")

    print("[PASS] Memory test complete")


def main():
    print("PDexFlow Forward Pass Test (RTX 3080 compatible)")
    print("=" * 60)

    try:
        test_forward_pass()
        test_ode_sampling()
        test_memory_usage()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nTo run inference with a real checkpoint:")
        print("  python -W ignore -m src.scripts.inference_pdexflow \\")
        print("      sample.ckpt_path=/path/to/checkpoint.pt \\")
        print("      sample.num_samples=1 \\")
        print("      sample.num_steps=30")

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
