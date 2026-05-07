# D-Peptide Structures (LoRA Fine-tuning)

Place your D-peptide PDB files here. Each file contains a single D-peptide structure.

```
data/d_pep_structures/
├── D_peptide_001.pdb
├── D_peptide_002.pdb
└── ...
```

## PDB Requirements

- Must contain N, CA, C backbone atoms
- D-amino acids use DAL, DGL, DPN, DTR, etc.
- Peptide length: ≥ 3 residues

## D-Amino Acid Naming Convention

| D-Amino Acid | PDB Name |
|-------------|----------|
| D-Ala       | DAL      |
| D-Arg       | DAR      |
| D-Asn       | DSN      |
| D-Asp       | DSP      |
| D-Cys       | DCY      |
| D-Gln       | DGN      |
| D-Glu       | DGL      |
| D-Phe       | DPN      |
| D-Gly       | GLY      |
| D-His       | DHI      |
| D-Ile       | DIL      |
| D-Leu       |DLE      |
| D-Lys       | DLY      |
| D-Met       | DME      |
| D-Pro       | DPRO     |
| D-Ser       | DSE      |
| D-Thr       | DTH      |
| D-Trp       | DTR      |
| D-Tyr       | DTY      |
| D-Val       | DVA      |

## Quick Data Generation

If you have L-peptide structures, you can mirror them to generate D-peptides:

```python
from src.data.parsers import parse_pdb

# Parse L-peptide
data, _ = parse_pdb('L_peptide.pdb')

# Mirror to D-peptide
data['pos_heavyatom'][..., 0] *= -1  # x-axis reflection

# Save as D-peptide PDB
# ...
```

## Reference

See D-Flow paper (https://arxiv.org/abs/2411.10618) for mirror-image algorithm.
