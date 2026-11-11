"""End-to-end smoke test of the real entry points on a tiny synthetic dataset (CPU, ~20 s):
toy data -> LMDB caches -> training -> Chiral-Bridge LoRA fine-tuning -> sampling -> evaluation."""
import glob
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SMALL = ["model.d_model=32", "model.n_layers=2", "model.n_heads=4", "model.max_len=64"]


def _run(args, cwd):
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONWARNINGS="ignore")
    proc = subprocess.run([sys.executable, "-W", "ignore", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"{' '.join(args)} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    return proc.stdout


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "configs", "pdexflow_training.yaml")), reason="configs missing")
def test_train_lora_sample_eval_pipeline(tmp_path):
    data = str(tmp_path / "pepmerge")
    cache = str(tmp_path / "cache")
    _run([os.path.join(ROOT, "scripts", "make_toy_dataset.py"), "--out", data], cwd=tmp_path)

    infer = [f"data.structure_dir={data}", f"data.cache_dir={cache}", *SMALL]
    common = [*infer, "train.device=cpu", "train.num_workers=0", "train.use_amp=false"]  # training configs only
    _run(["-m", "src.cli.train", *common, "train.batch_size=4", "train.max_iters=6", "train.save_freq=3",
          f"train.logdir={tmp_path / 'logs'}"], cwd=tmp_path)
    ckpt = glob.glob(str(tmp_path / "outputs" / "checkpoints" / "*" / "model" / "best_model.pt"))
    assert ckpt, "training produced no checkpoint"

    _run(["-m", "src.cli.finetune", *common, "train.batch_size=4", "train.max_iters=4", "train.save_freq=2",
          f"train.logdir={tmp_path / 'lora_logs'}", f"sample.ckpt_path={ckpt[0]}"], cwd=tmp_path)
    lora = glob.glob(str(tmp_path / "lora_logs" / "checkpoints" / "*" / "model" / "best_model.pt"))
    assert lora, "LoRA fine-tuning produced no checkpoint"

    for token in (0, 1):
        out = _run(["-m", "src.cli.sample", *infer, "sample.device=cpu", f"sample.ckpt_path={ckpt[0]}",
                    f"sample.lora_ckpt_path={lora[0]}", "chi_lora.enable=true", "sample.num_samples=2", "sample.num_steps=3",
                    f"sample.chirality_token={token}", f"sample.output_dir={tmp_path / f'peptides{token}'}"], cwd=tmp_path)
        assert glob.glob(str(tmp_path / f"peptides{token}" / "*" / "*" / "sample_0.pdb")), "no sampled structure written"

    out = _run(["-m", "src.cli.evaluate", *infer, "sample.device=cpu", f"sample.ckpt_path={ckpt[0]}",
                "sample.max_eval=2", "sample.num_steps=3", f"sample.output_dir={tmp_path / 'eval'}", "eval.dataset_x_mirror=true",
                "eval.reference.enable=true", "eval.reference.max_samples=4"], cwd=tmp_path)  # also exercises the novelty path
    metrics = dict(line.split(": ", 1) for line in out.splitlines() if ": " in line and line.split(": ", 1)[0].replace("_", "").isalnum())
    assert "novelty_seq_id_max_mean" in metrics and metrics["novelty_seq_id_max_mean"] != "nan", "reference bank was not used"
    assert "aar_mean" in metrics and float(metrics["chiral_error_mean"]) == 0.0, "D-mode evaluation must report 0 chirality error"
