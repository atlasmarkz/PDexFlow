# PDexFlow (UniChiFlow)

**PDexFlow: Parity-Equivariant Flow Matching for D-Peptide Design**

PDexFlow is a state-of-the-art flow matching model designed for generating **D-peptides** (and mixed-chirality peptides) with high geometric validity and biological plausibility. It overcomes the limitations of standard generative models by enforcing physical symmetry constraints and utilizing a novel geometric decoding strategy.

## Key Features

*   **Parity-Harmonized Flow Matching (PH-Flow)**: Learns a vector field equivariant to coordinate reflections, enabling robust learning from limited D-peptide data by leveraging the vast L-peptide dataspace.
*   **Chiral-Bridge LoRA (ChiLoRA)**: A parameter-efficient, chirality-gated adapter that specializes the backbone Model for specific chiral behaviors without catastrophic forgetting.
*   **Hard Geometric Decoder (Reconstruct-then-Reflect)**: A novel decoding strategy that guarantees **0.00% Chiral Error**. It reconstructs residues using standard L-geometry and applies a global coordinate reflection for D-residues, ensuring perfect sidechain-backbone consistency.
*   **Minibatch Optimal Transport (OT)**: Aligns training noise with data to create straight flow trajectories, accelerating convergence and inference.

## Environment
```bash
cd /path/to/UniChiflow
conda create -n unichiflow python=3.10
conda activate unichiflow
pip install -r requirements.txt
```

## Installation

```bash
cd /path/to/UniChiflow
pip install -e .
```

*Note: Evaluation requires [PyRosetta](https://www.pyrosetta.org/) installed in the environment.*

## Workflow

### 1. Training

PDexFlow supports two-stage training (Pre-training on L-peptides -> Fine-tuning on D-peptides with LoRA) or direct training.

**Fine-tuning with LoRA (recommended for D-peptides):**
```bash
python -W ignore -m unichiflow.experiments.train_lora_real_d
```
*Config Path: `unichiflow/configs/unichiflow_lora.yaml`*

**Standard Training:**
```bash
python -W ignore -m unichiflow.experiments.train_unichiflow
```

### 2. Inference (Generation)

Generate D-peptide candidates for a given receptor target. The model outputs `.pdb` files with correct D-amino acid nomenclature (DAL, DGL, etc.).

```bash
python -W ignore -m unichiflow.experiments.inference_unichiflow \
    sample.num_samples=100 \
    sample.ckpt_path=path/to/checkpoint.pt
```

### 3. Evaluation

We provide a comprehensive PyRosetta-based evaluation suite (`Eval/`) that measures geometric validity, biological plausibility, and binding affinity. The evaluation pipeline automatically handles D-peptide flipping to ensure accurate energy calculation in standard forcefields.

```bash
# Run batch evaluation on a folder of generated PDBs
python Eval/batch_eval.py /path/to/generated_pdbs \
    --output metrics_summary.csv \
    --workers 8
```

#### Metrics Explained:
*   **AAR (Atom-Atom Recovery)**: Contact recovery rate against reference.
*   **RMSD**: Geometric deviation of the backbone.
*   **SCR (Sidechain Recovery)**: Percentage of sidechains with correct $\chi$ angles (New!).
*   **Chiral_Rec / Chiral_pct**: Chiral correctness. **PDexFlow achieves ~100% Chiral Recovery** due to the hard decoder.
*   **Stb (Stability)**: Rosetta energy-based stability score (calculated in standardized L-space).
*   **Aff (Affinity)**: Interaction energy ($\Delta G$) with the receptor.

## Ablations & Configurations

You can toggle core components via Hydra CLI flags:

- **Disable Symmetry Loss**: `flow.sym_weight=0.0`
- **Disable Chiral-Bridge**: `chi_lora.enable=false esm.use_adapter=false`
- **Disable OT**: `flow.use_ot=false`
- **Disable Hard Decoder (Ablation testing)**: `sample.use_hard_decoder=false` (Note: This will result in high Chiral Error).

## Folder Structure

*   `unichiflow/models`: Core architecture (`chiralformer.py`), ChiLoRA, and Flow logic.
*   `unichiflow/data`: Geometry handling, torsion conversion (`torsion.py`), and constants.
*   `Eval/`: Evaluation scripts.
    *   `pyrosetta_metrics.py`: Metrics implementation + D-peptide flipping logic.
    *   `batch_eval.py`: Parallel processing script.
