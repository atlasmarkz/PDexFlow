<h1><img src="assets/logo.png" alt="PDexFlow logo" width="72" align="absmiddle"> PDexFlow</h1>

**Escaping the Mirror Trap: Parity-Harmonized Flow Matching on Riemannian Manifolds for Peptide Design** — generate L- and D-peptide binders with symmetry-consistent flow matching and hard geometric constraints.

Official code for the paper accepted at the *International Conference on Data Mining (ICDM) 2026*.

<p align="center">
  <img src="assets/architecture.png" alt="PDexFlow architecture" width="900">
</p>

## Installation

```bash
conda create -n pdexflow python=3.10
conda activate pdexflow
pip install -r requirements.txt
pip install -e .
```

## Usage

Hydra configs are read from `configs/` (local, not tracked in this repository).

**Train**

```bash
python -W ignore -m src.experiments.train_pdexflow \
    --config-path configs --config-name unichiflow_pure_L
```

**Generate** (`chirality=0` → L-peptide, `chirality=1` → D-peptide)

```bash
python -W ignore scripts/inference_pdexflow.py \
    --config-path configs --config-name unichiflow_inference \
    sample.num_samples=100 sample.ckpt_path=/path/to/checkpoint.pt
```

**Evaluate**

```bash
python -W ignore scripts/eval_pdexflow.py \
    --config-path configs --config-name unichiflow_eval \
    sample.ckpt_path=/path/to/checkpoint.pt
```

**Smoke test (CPU, no dataset needed)**

```bash
python scripts/smoke_cpu.py
```
