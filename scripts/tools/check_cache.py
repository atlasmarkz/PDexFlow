"""Check LMDB cache for corrupted samples."""
import os
import pickle
import lmdb
import torch
from glob import glob

def check_lmdb_cache(cache_path):
    """Check all samples in LMDB cache for invalid geometry."""
    if not os.path.exists(cache_path):
        print(f"Cache not found: {cache_path}")
        return

    db = lmdb.open(cache_path, readonly=True, subdir=False, lock=False, readahead=False, meminit=False)
    issues = []

    with db.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            sample_id = key.decode('utf-8')
            try:
                data = pickle.loads(value)

                # Check pos_heavyatom
                if not torch.all(torch.isfinite(data['pos_heavyatom'])):
                    issues.append(f"{sample_id}: pos_heavyatom has NaN/Inf")

                # Check torsion_angle
                if not torch.all(torch.isfinite(data['torsion_angle'])):
                    issues.append(f"{sample_id}: torsion_angle has NaN/Inf")

                # Check generate_mask length
                pep_len = data['generate_mask'].sum().item()
                if pep_len < 3:
                    issues.append(f"{sample_id}: peptide length {pep_len} < 3")
                if pep_len > 25:
                    issues.append(f"{sample_id}: peptide length {pep_len} > 25")

                # Check aa values
                aa = data['aa']
                if (aa >= 20).any():
                    issues.append(f"{sample_id}: contains UNK residues (aa >= 20)")

            except Exception as e:
                issues.append(f"{sample_id}: failed to load - {e}")

    db.close()

    if issues:
        print(f"Found {len(issues)} problematic samples:")
        for issue in issues[:50]:  # Show first 50
            print(f"  {issue}")
        if len(issues) > 50:
            print(f"  ... and {len(issues) - 50} more")
    else:
        print(f"Cache OK: all {cache_path} samples are valid")


if __name__ == "__main__":
    import sys
    cache_paths = [
        ".cache/pep_cache_lora_d",
        ".cache/pep_cache",
    ]
    for cp in cache_paths:
        print(f"\nChecking {cp}...")
        check_lmdb_cache(cp)