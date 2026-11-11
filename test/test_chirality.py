
import torch
import torch.nn.functional as F
import math

# Mocking the necessary parts to test decode_to_heavyatom15
from src.models.chiralformer import PDexFlowModel
from src.data import pep_constants as constants
from src.data.torsion import full_atom_reconstruction

def test_decoder_chirality():
    # Setup
    B, L = 1, 5
    # AA: ALA (index 0)
    aa = torch.zeros((B, L), dtype=torch.long) # ALA
    
    # Random trans/rot/angles
    trans = torch.randn((B, L, 3))
    rotmats = torch.eye(3).view(1, 1, 3, 3).repeat(B, L, 1, 1)
    angles = torch.randn((B, L, 5))
    
    # Chirality = 1 (D)
    chirality = torch.ones((B,), dtype=torch.long)
    peptide_mask = torch.ones((B, L), dtype=torch.long)
    
    model = PDexFlowModel() # Init model (weights don't matter for decode)
    
    # Decode
    pos15, mask15 = model.decode_to_heavyatom15(
        aa=aa,
        trans=trans,
        rotmats=rotmats,
        angles=angles,
        chirality=chirality,
        peptide_mask=peptide_mask,
        hard_constraints=True
    )
    
    # Check Chirality of Residue 0
    # ALA: N, CA, C, CB
    # We need to manually check the geometry of the Output
    # But wait, pos15 is (B, L, 15, 3)
    
    # Get atoms for first residue (ALA)
    # Mapping: N=0, CA=1, C=2, CB=4 (for ALA? Need to check constants)
    # Check constants.restype_name_to_atom14_names['ALA']
    
    print("Decoding complete.")
    
    # Let's inspect coordinates of first residue
    res_pos = pos15[0, 0] # (15, 3)
    
    # N, CA, C, CB
    # Assuming standard indexing: N=0, CA=1, C=2
    # CB index is variable. For ALA it's usually 4 (after O). 
    # Let's check dists to confirm connectivity
    n = res_pos[0]
    ca = res_pos[1]
    c = res_pos[2]
    # Find CB: non-zero
    # ALA has CB.
    
    # Calculate geometric chirality
    # v1 = N - CA
    # v2 = C - CA
    # v3 = CB - CA
    # Vol = (v1 x v2) . v3
    
    # We need CB index. From torsion.py/constants, usually:
    # N, CA, C, O, CB ...
    cb = res_pos[4] 
    
    v1 = n - ca
    v2 = c - ca
    v3 = cb - ca
    
    cross = torch.cross(v1, v2)
    vol = (cross * v3).sum()
    
    print(f"Chiral Volume: {vol.item()}")
    
    # L-amino acid: Volume is positive (usually ~2.0 depending on ordering)
    # D-amino acid: Volume is negative
    
    if vol.item() < -0.1:
        print("SUCCESS: Generated D-Amino Acid (Negative Volume)")
    elif vol.item() > 0.1:
        print("FAILURE: Generated L-Amino Acid (Positive Volume)")
    else:
        print("Ambiguous geometry")

if __name__ == "__main__":
    test_decoder_chirality()
