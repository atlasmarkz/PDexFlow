# UniChiflow (PDexFlow)

**UniChiflow: Parity-Equivariant Flow Matching for D-Peptide Design**

UniChiflow is a state-of-the-art flow matching model designed for generating **D-peptides** (and mixed-chirality peptides) with high geometric validity and biological plausibility. It overcomes the limitations of standard generative models by enforcing physical symmetry constraints and utilizing a novel geometric decoding strategy.

## Key Features

*   **Parity-Harmonized Flow Matching (PH-Flow)**: Learns a vector field equivariant to coordinate reflections, enabling robust learning from limited D-peptide data by leveraging the vast L-peptide dataspace.
*   **Chiral-Bridge LoRA (ChiLoRA)**: A parameter-efficient, chirality-gated adapter that specializes the backbone Model for specific chiral behaviors without catastrophic forgetting.
*   **Hard Geometric Decoder (Reconstruct-then-Reflect)**: A novel decoding strategy that guarantees **0.00% Chiral Error**. It reconstructs residues using standard L-geometry and applies a global coordinate reflection for D-residues, ensuring perfect sidechain-backbone consistency.
*   **Minibatch Optimal Transport (OT)**: Aligns training noise with data to create straight flow trajectories, accelerating convergence and inference.

## Environment

```bash
conda create -n unichiflow python=3.10
conda activate unichiflow
pip install -r requirements.txt
```

## Installation

```bash
pip install -e .
```

*Note: Evaluation requires [PyRosetta](https://www.pyrosetta.org/) installed in the environment.*

## Project Structure

```
unidexflow/
├── src/                  # Core package
│   ├── data/             # Data loading, geometry, torsion conversion
│   ├── models/           # Model architecture (ChiralFormer, ChiLoRA)
│   ├── analysis/        # Analysis utilities and metrics
│   └── flow.py           # Core flow matching logic
├── configs/              # Hydra configuration templates (*.example.yaml)
├── experiments/          # Training scripts
├── scripts/              # Inference and evaluation scripts
├── tests/                # Unit tests
└── outputs/              # Training outputs (gitignored)
```

## Configuration

Configs are templates with `.example.yaml` suffix. Copy to `config.yaml` and modify paths as needed:

```bash
# Create local config
cp configs/unichiflow_backbone.example.yaml configs/config.yaml

# Edit paths (data directory, checkpoint paths, etc.)
# configs/config.yaml is gitignored - do not commit your local overrides
```

**Default paths in example configs:**
- `data/` — Place preprocessed PDB structures here
- `.cache/` — LMDB cache for preprocessed data (auto-created on first run)
- `outputs/` — Training logs and checkpoints

## Workflow

### 1. Training

UniChiflow supports two-stage training (Pre-training on L-peptides -> Fine-tuning on D-peptides with LoRA) or direct training.

**Fine-tuning with LoRA (recommended for D-peptides):**
```bash
python -W ignore -m src.experiments.train_lora_real_d \
    --config-path configs \
    --config-name unichiflow_lora
```

**Standard Training:**
```bash
python -W ignore -m src.experiments.train_unichiflow \
    --config-path configs \
    --config-name unichiflow_backbone
```

Override paths via CLI:
```bash
python -W ignore -m src.experiments.train_unichiflow \
    data.structure_dir=/path/to/structures \
    sample.ckpt_path=/path/to/checkpoint.pt
```

### 2. Inference (Generation)

Generate D-peptide candidates for a given receptor target. The model outputs `.pdb` files with correct D-amino acid nomenclature (DAL, DGL, etc.).

```bash
python -W ignore -m src.scripts.inference_unichiflow \
    --config-path configs \
    --config-name unichiflow_inference \
    sample.num_samples=100 \
    sample.ckpt_path=/path/to/checkpoint.pt
```

### 3. Evaluation

We provide a comprehensive PyRosetta-based evaluation suite that measures geometric validity, biological plausibility, and binding affinity. The evaluation pipeline automatically handles D-peptide flipping to ensure accurate energy calculation in standard forcefields.

```bash
python -W ignore -m src.scripts.eval_unichiflow \
    --config-path configs \
    --config-name unichiflow_eval \
    sample.ckpt_path=/path/to/checkpoint.pt
```

Optional robustness summary from an ablation CSV:
```bash
python -W ignore -m src.scripts.summarize_sensitivity \
    path/to/ablation_runs.csv \
    --metrics chiral_err,rmsd_ca,motif_acc
```

#### Metrics Explained:
*   **AAR (Atom-Atom Recovery)**: Contact recovery rate against reference.
*   **RMSD**: Geometric deviation of the backbone.
*   **SCR (Sidechain Recovery)**: Percentage of sidechains with correct χ angles.
*   **Chiral_Rec / Chiral_pct**: Chiral correctness. **UniChiflow achieves ~100% Chiral Recovery** due to the hard decoder.
*   **Stb (Stability)**: Rosetta energy-based stability score (calculated in standardized L-space).
*   **Aff (Affinity)**: Interaction energy (ΔG) with the receptor.

## Ablations & Configurations

You can toggle core components via Hydra CLI flags:

- **Disable Symmetry Loss**: `flow.sym_weight=0.0`
- **Disable Chiral-Bridge**: `chi_lora.enable=false esm.use_adapter=false`
- **Disable OT**: `flow.use_ot=false`
- **Disable Hard Decoder (Ablation testing)**: `sample.use_hard_decoder=false` (Note: This will result in high Chiral Error).

## Code Quality

This project uses ruff for linting and mypy for type checking:

```bash
# Lint
ruff check .

# Format
ruff format .

# Type check
mypy src/
```

## Data & Artifacts

- **Metrics** (`metrics/`) — Experiment result CSVs, gitignored. Add your own results for local reference.
- **Checkpoints** — Saved to `outputs/` or `outputs_lora/`, gitignored.
- **Cache** (`.cache/`) — LMDB databases auto-generated from PDB structures, gitignored.

## License

See LICENSE file.
