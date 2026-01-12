# UniChiFlow (scaffold)

This folder contains an implementation of **UniChiFlow** described in `UniChiflow.md`.

- Mirrors the high-level structure of `PeptideDesign/` (configs/data/models/experiments).
- **ChiLoRA / Chiral-Bridge is now implemented** with optional chirality-gated LoRA adapters; enable via config (see below).
- **Minibatch OT alignment is now configurable** (`flow.use_ot: true|false`) to toggle OT-based permutation alignment during training.
- **Ablations are one-flag away**: symmetry loss (`flow.sym_weight`), Chiral-Bridge (`chi_lora.enable` + `esm.use_adapter`), hard constraints (`sample.use_hard_decoder`), ROT (`flow.use_ot`).
- Uses **PepMerge** as the raw training/testing source: `/root/autodl-tmp/pepmerge`.

## Quick start

```bash
cd /root/autodl-fs/UniChiflow
pip install -e .

# Train (sanity)
python -W ignore -m unichiflow.experiments.train_unichiflow

# Inference (writes PDBs)
python -W ignore -m unichiflow.experiments.inference_unichiflow
```

Config is in `unichiflow/configs/unichiflow_pepmerge.yaml`.

### OT alignment toggle
- Set `flow.use_ot: true` (default) to enable minibatch OT permutation alignment between noisy and data coordinates during training.
- Set to `false` to disable OT (use plain random noise without alignment).

### Quick ablation overrides (Hydra)
- w/o Symmetry: `python -m unichiflow.experiments.train_unichiflow flow.sym_weight=0.0`
- w/o Chiral-Bridge: `python -m unichiflow.experiments.train_unichiflow chi_lora.enable=false esm.enable=true esm.use_adapter=false`
- w/o Hard Constraints (inference): `python -m unichiflow.experiments.inference_unichiflow sample.use_hard_decoder=false`
- w/o ROT: `python -m unichiflow.experiments.train_unichiflow flow.use_ot=false`
- Gradient accumulation (to keep大batch等效): `train.grad_accum_steps=<N>`

### Evaluate the ablations
Run inference-side metrics (Motif Recovery, Chiral Error, scRMSD, Conv. NFE):
```bash
python -m unichiflow.experiments.eval_unichiflow \
	sample.ckpt_path=/path/to/ckpt.pt \
	sample.num_steps=30 sample.max_eval=32
```
Metrics会打印到stdout，并在`sample.output_dir`下保存`pred_eval.pdb`与`gt_eval.pdb`示例。

## ChiLoRA (Chiral-Bridge)

- Toggle: set `chi_lora.enable: true` in `unichiflow/configs/unichiflow_pepmerge.yaml`.
- Targets: choose in `chi_lora.targets` (`attn_out`, `ffn`, or both).
- Tuning: adjust `rank`, `alpha`, `dropout`; set `freeze_backbone: true` to train LoRA-only.
- Gating: LoRA is automatically gated by peptide chirality tokens (active on D residues only).
