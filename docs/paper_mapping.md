# Paper ↔ code mapping

Reference: *Escaping the Mirror Trap: Parity-Harmonized Flow Matching on Riemannian Manifolds for Peptide Design* (ICDM 2026).
Status of each item was checked against the code; "tested" means covered by `test/`.

| Paper | Code | Status |
|---|---|---|
| Parity operator `P(R,x) = (-R·I_flip, -x)`, `I_flip = diag(1,1,-1)` (Sec. III-C) | `src/data/parity.py` (`parity_positions`, `parity_rotations` = `R·diag(-1,-1,1)`, `parity_torsions`, tangent actions) | tested: frames/torsions measured from inverted coordinates equal the parity action |
| Mirror-symmetric training data (Sec. IV-A) | `scripts/generate_mirrored_dataset.py`, `TwinUniversePepDataset`, `DualPepDataset` (spatial inversion) | implemented |
| Geodesic probability path (Eq. 3) | `PDexFlowModel.corrupt_batch`: linear (translation), SO(3) geodesic, torus geodesic | implemented |
| `L_flow` (Eq. 4) | `src/flow.py::flow_matching_loss` | `flow.loss_type=l2` is the paper's squared norm; **default `huber`** (robust against the `1/(1-t)` velocity amplification) |
| Symmetry Consistency Loss `L_sym` (Eq. 5): two passes `(z,c)` and `(P z, 1-c)` | `src/flow.py::compute_symmetry_loss` | tested: zero for an equivariant field, positive otherwise |
| Mini-ROT (Eq. 13): assignment between the `B` priors and `B` data samples | `src/data/minibatch_ot.py::minibatch_ot_pair`, enabled by `flow.use_ot` | tested, parity-invariant cost |
| Input featurization (Eq. 9) | `PDexFlowModel.predict_x1` | implemented (plus position / generate-mask embeddings) |
| Geometric Transformer: RBF distance features, peptide self-attention + receptor cross-attention (Sec. IV-C) | `src/models/geometric_transformer.py` | tested; the old plain-transformer trunk, its checkpoints and Axial Feature Injection were removed |
| Chiral-Bridge LoRA `y = W0 x + G(c)·α/r·(BA)x`, `G(c) = c·σ(γ)` (Eq. 7-8) on q/k/v/out and FFN | `src/models/chiral_gated_lora.py`, `PDexFlowModel._apply_chi_lora` (`chi_lora.targets=[attn, ffn]`) | tested: inactive for `c=0`, active for `c=1`; `γ` stays trainable with a frozen backbone |
| Frozen ESM-2 receptor features | `src/models/semantic_conditioner.py` (`esm.enable`) | receptor-only tokens are encoded and added to receptor tokens; **not covered by tests** (needs `fair-esm` and weights); default model is `esm2_t6_8M` |
| Chirality-preserving NeRF decoder `X = R·Q_c·NeRF(a,χ) + x`, `Q_c = diag(1,1,-1)` for D (Eq. 10-11) | `src/models/hard_geometric_decoder.py` | tested: `decode(measure(D structure))` reproduces the structure to <1e-2 Å; D-mode sampling gives 0 % chirality error, switching the decoder off gives 100 % |
| `L_total = L_flow + λ_sym L_sym + λ_seq L_seq` (Eq. 12) | `PDexFlowModel._compute_total_loss` | implemented (`flow.sym_weight`, `flow.seq_weight`); optional sym warm-up and Kendall weighting are extras |
| Sampling: integrate `dz/dt = v_θ` (Eq. 14) | `PDexFlowModel.sample` | `x_1`-prediction Euler steps `α = Δt/(1-t)`, which equals the ODE Euler step on the geodesic path; sequence tokens are committed following the masking schedule |
| Ablations (w/o `L_sym`, Chiral-Bridge, Hard Constraint, ROT) | `flow.sym_weight=0`, `chi_lora.enable=false`, `sample.use_hard_decoder=false`, `flow.use_ot=false` | available |

## Not part of the paper
* Auxiliary sequence masking (peptide tokens are masked with probability `1-t`) is an implementation detail of the sequence head.

## Evaluation metrics
`scripts/eval_pdexflow.py` reports **AAR** (Amino Acid Recovery), CA-RMSD, chirality error (generated residues that own a CB
atom), novelty, diversity (clustering of CA descriptors) and latency / memory. The paper's **Clash, SSR, BSR, scRMSD,
Stb. and Aff.** are *not* computed by this repository. Reproducing them fully requires a separately installed PyRosetta (separate
licence, not a pip dependency); Rosetta based numbers can be supplied through external CSVs
(`eval.orthogonal`), the others require external tools (DSSP, a structure predictor, Rosetta) and are not implemented.

## Bugs fixed while aligning with the paper
* `src/data/parity.py` was imported but never committed (a `data/` rule in `.gitignore` also matched `src/data/`).
* The mirrored dataset used `(x, y, z) → (-x, -y, z)`, a rotation by π (no change of handedness) instead of an inversion.
* `so3.log_map` returned twice the rotation angle, which doubled every rotation velocity and geodesic distance.
* The torsion `psi` of the data pipeline and of the rigid-group reconstruction differed by π (backbone O was misplaced by ~2.1 Å).
* The Chiral-Bridge gate was never applied when AFI was enabled, and the legacy trunk's fast path bypassed LoRA in eval mode.
* `chirality_error` scored receptor and glycine residues and broadcast the chirality vector incorrectly for batch size > 1.
* Inference/evaluation built the model differently from training; all entry points now share `build_model`.
