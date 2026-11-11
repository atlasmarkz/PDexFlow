<h1><img src="assets/logo.svg" alt="PDexFlow logo" width="72" align="absmiddle"> PDexFlow</h1>

**Parity-Harmonized Flow Matching for De Novo D-Peptide Design**

This repository (`PDexFlow`) is the official implementation of **PDexFlow**, the framework proposed in the paper *Parity-Harmonized Flow Matching for De Novo D-Peptide Design* (PDF included). The Python package is named `pdexflow`; the main model class is `PDexFlowModel` (the former name `UniChiFlowModel` remains available as an alias).

PDexFlow is a flow matching model designed for generating **D-peptides** (and mixed-chirality peptides) with high geometric validity and biological plausibility. It overcomes the limitations of standard generative models by enforcing physical symmetry constraints and utilizing a novel geometric decoding strategy.

## Key Features

*   **Parity-Harmonized Flow Matching (PH-Flow)**: Learns a vector field equivariant to coordinate reflections, enabling robust learning from limited D-peptide data by leveraging the vast L-peptide dataspace.
*   **Axial Feature Injection (AFI)**: Breaks E(3) chirality blindness by injecting parity-invariant axial features (cross products, commutators), stabilized with sym warmup.
*   **Chiral-Bridge LoRA (ChiLoRA)**: A parameter-efficient, chirality-gated adapter that specializes the backbone model for specific chiral behaviors without catastrophic forgetting.
*   **Hard Geometric Decoder (Reconstruct-then-Reflect)**: A novel decoding strategy that guarantees **0.00% Chiral Error**. It reconstructs residues using standard L-geometry and applies a global coordinate reflection for D-residues, ensuring perfect sidechain-backbone consistency.
*   **Minibatch Optimal Transport (OT)**: Aligns training noise with data to create straight flow trajectories, accelerating convergence and inference.

## Environment

```bash
conda create -n pdexflow python=3.10
conda activate pdexflow
pip install -r requirements.txt
```

## Installation

```bash
pip install -e .
```

*Note: Evaluation requires [PyRosetta](https://www.pyrosetta.org/) installed in the environment.*

## Project Structure

```
PDexFlow/
├── src/                  # Core package
│   ├── data/             # Data loading, geometry, torsion, parity
│   ├── models/           # Model architecture (ChiralFormer, ChiLoRA, AFI)
│   ├── analysis/         # Analysis utilities and metrics
│   ├── experiments/       # Training scripts and components
│   └── flow.py           # Core flow matching logic (PH-Flow)
├── configs/              # Hydra configuration templates
├── scripts/              # Inference and evaluation scripts
│   ├── analyze_dataset.py       # Dataset distribution analysis
│   ├── generate_mirrored_dataset.py  # Generate mirrored D-peptide dataset
│   └── plot_ramachandran_from_npz.py  # Ramachandran plot from cached data
├── tests/                # Unit tests
└── outputs/              # Training outputs (gitignored)
```

## Configuration

Configs are in `configs/` directory. Copy template and modify paths as needed:

```bash
# Create local config
cp configs/unichiflow_training.yaml configs/config.yaml

# Edit paths (data directory, checkpoint paths, etc.)
# configs/config.yaml is gitignored - do not commit your local overrides
```

**Available configs:**
- `unichiflow_pure_L.yaml` — Pure L-peptide training (recommended, leverages sym loss for D-peptide generation)
- `unichiflow_training.yaml` — Standard training with TwinUniverseDataset
- `unichiflow_lora.yaml` — LoRA fine-tuning for D-peptides
- `unichiflow_inference.yaml` — Inference configuration

**Default paths in configs:**
- `data/` — Place preprocessed PDB structures here
- `.cache/` — LMDB cache for preprocessed data (auto-created on first run)
- `outputs/` — Training logs and checkpoints

## Workflow

### 1. Training

PDexFlow supports pure L-peptide training with PH-Flow symmetry loss sufficient for generating both L and D peptides.

**Pure L Training (Recommended):**
```bash
python -W ignore -m src.experiments.train_pdexflow \
    --config-path configs \
    --config-name unichiflow_pure_L
```

**Standard Training (with TwinUniverseDataset):**
```bash
python -W ignore -m src.experiments.train_pdexflow \
    --config-path configs \
    --config-name unichiflow_training
```

**Fine-tuning with LoRA (for D-peptides):**
```bash
python -W ignore -m src.experiments.train_lora_real_d \
    --config-path configs \
    --config-name unichiflow_lora
```

### 2. Generate Mirrored D-Peptide Dataset (Optional)

Generate parity-mirrored (x, y, z) → (-x, -y, z) D-peptide dataset for comparative training:

```bash
python scripts/generate_mirrored_dataset.py \
    --input_dir data/pepmerge \
    --output_dir data/pepmerge_mirrored
```

This creates a separate mirrored dataset for pure L vs pure D training comparison.

### 3. Inference (Generation)

Generate D-peptide candidates for a given receptor target. The model outputs `.pdb` files with correct D-amino acid nomenclature (DAL, DGL, etc.).

```bash
python -W ignore scripts/inference_pdexflow.py \
    --config-path configs \
    --config-name unichiflow_inference \
    sample.num_samples=100 \
    sample.ckpt_path=/path/to/checkpoint.pt
```

**Chirality Control at Inference:**
- `chirality=0` → generates L-peptide
- `chirality=1` → generates D-peptide (via Hard Geometric Decoder mirror)

### 4. Evaluation

We provide a comprehensive PyRosetta-based evaluation suite that measures geometric validity, biological plausibility, and binding affinity. The evaluation pipeline automatically handles D-peptide flipping to ensure accurate energy calculation in standard forcefields.

```bash
python -W ignore scripts/eval_pdexflow.py \
    --config-path configs \
    --config-name unichiflow_eval \
    sample.ckpt_path=/path/to/checkpoint.pt
```

Optional robustness summary from an ablation CSV:
```bash
python -W ignore scripts/summarize_sensitivity.py \
    path/to/ablation_runs.csv \
    --metrics chiral_err,rmsd_ca,motif_acc
```

#### Metrics Explained:
*   **AAR (Atom-Atom Recovery)**: Contact recovery rate against reference.
*   **RMSD**: Geometric deviation of the backbone.
*   **SCR (Sidechain Recovery)**: Percentage of sidechains with correct χ angles.
*   **Chiral_Rec / Chiral_pct**: Chiral correctness. **PDexFlow achieves ~100% Chiral Recovery** due to the hard decoder.
*   **Stb (Stability)**: Rosetta energy-based stability score (calculated in standardized L-space).
*   **Aff (Affinity)**: Interaction energy (ΔG) with the receptor.

## Data Analysis

Analyze dataset distribution (length, amino acid composition, Ramachandran plots, etc.):

```bash
# Full analysis
python scripts/analyze_dataset.py --max_samples 10348

# Quick analysis with fewer samples
python scripts/analyze_dataset.py --max_samples 500

# Plot Ramachandran from cached npz (faster)
python scripts/plot_ramachandran_from_npz.py
```

Outputs are saved to `outputs/analysis/` with:
- `peptide/` - Peptide statistics and plots
- `receptor/` - Receptor statistics and plots
- `dataset_distribution_data.npz` - Cached raw data for fast replotting

## Ablations & Configurations

You can toggle core components via Hydra CLI flags:

- **Disable Symmetry Loss**: `flow.sym_weight=0.0`
- **Disable Sym Warmup**: `flow.sym_warmup_enable=false`
- **Disable AFI**: `afi.enable=false`
- **Disable Chiral-Bridge**: `chi_lora.enable=false esm.enable=false`
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

## Training Health Monitoring

Monitor these metrics during training to diagnose issues:

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| `train/skip_rate` | <2% | 2-10% | >10% |
| `train/bad_step_streak` | <3 | 3-5 | ≥5 |
| `train/aa_accuracy` | >30% | 15-30% | <15% |

- **skip_rate**: Percentage of steps skipped due to non-finite loss/gradients
- **bad_step_streak**: Consecutive skipped steps (indicates systematic issues)
- **aa_accuracy**: Amino acid sequence recovery (AAR)

## Data & Artifacts

- **Metrics** (`metrics/`) — Experiment result CSVs, gitignored. Add your own results for local reference.
- **Checkpoints** — Saved to `outputs/<timestamp>/checkpoints/`, gitignored.
- **Cache** (`.cache/`) — LMDB databases auto-generated from PDB structures, gitignored.
- **Mirrored Dataset** (`data/pepmerge_mirrored/`) — parity-mirrored D-peptide data, gitignored.

## Training Strategy Explanation

### PH-Flow Symmetry Loss

The symmetry consistency loss (`sym_loss`) in PH-Flow enforces parity equivariance:

```
sym_loss: P(v(x_t)) = v(P(x_t))
```

Where `P` is the parity transformation (position flip, rotation mirror, angle negation). This allows the model to learn both L and D generation from pure L training data.

### Axial Feature Injection (AFI)

AFI breaks E(3) chirality blindness by injecting parity-invariant axial features:

- **Cross products**: capture right-hand rule chirality
- **Commutator features**: detect handedness violations
- **Triple product projections**: invariant under parity

AFI gate_logit initializes at -2.0 (sigmoid≈0.12), allowing gradual activation during sym warmup.

### Hard Geometric Decoder

The decoder applies chirality transformation during coordinate reconstruction:
- For D residues (`chirality=1`): `rot @ diag(-1,-1,1)` and `angles * -1`
- This guarantees **0% chiral error** by construction

## License

See LICENSE file.