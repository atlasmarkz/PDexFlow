"""Salvage readable entries from a corrupted LMDB cache."""
import os
import pickle
import lmdb
import torch

def diagnose_lmdb(cache_path):
    """Check what's inside the LMDB."""
    lmdb_file = [f for f in os.listdir(cache_path) if f.endswith('.lmdb') and not f.endswith('-lock')]
    if not lmdb_file:
        print(f"No LMDB file found in {cache_path}")
        return

    lmdb_path = os.path.join(cache_path, lmdb_file[0])
    print(f"Checking {lmdb_path}")

    env = lmdb.open(lmdb_path, readonly=True, subdir=False, lock=False, readahead=False, meminit=False)
    stats = env.stat()
    print(f"Entries reported by stat: {stats['entries']}")

    issues = []
    valid_count = 0
    read_errors = 0

    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            for i, (key, value) in enumerate(cursor):
                sample_id = key.decode('utf-8')
                try:
                    data = pickle.loads(value)

                    # Check pos_heavyatom
                    if not torch.all(torch.isfinite(data['pos_heavyatom'])):
                        issues.append(f"{sample_id}: pos_heavyatom NaN/Inf")
                        continue

                    # Check torsion_angle
                    if not torch.all(torch.isfinite(data['torsion_angle'])):
                        issues.append(f"{sample_id}: torsion_angle NaN/Inf")
                        continue

                    # Check peptide length
                    pep_len = data['generate_mask'].sum().item()
                    if pep_len < 3 or pep_len > 25:
                        issues.append(f"{sample_id}: pep_len={pep_len}")
                        continue

                    # Check aa values
                    aa = data['aa']
                    if (aa >= 20).any():
                        issues.append(f"{sample_id}: UNK residues")
                        continue

                    valid_count += 1

                except Exception as e:
                    read_errors += 1
                    issues.append(f"{sample_id}: {type(e).__name__}: {e}")

                if i % 10000 == 0 and i > 0:
                    print(f"  Processed {i} entries...")

    except Exception as e:
        print(f"Error during iteration: {e}")

    env.close()

    print(f"\nValid samples: {valid_count}")
    print(f"Read errors: {read_errors}")
    print(f"Validation issues: {len(issues)}")

    if issues:
        print("\nFirst 20 issues:")
        for issue in issues[:20]:
            print(f"  {issue}")


if __name__ == "__main__":
    caches = [
        ".cache/pep_cache",
        ".cache/pep_cache_pure_L",
    ]
    for cache in caches:
        if os.path.exists(cache):
            print(f"\n{'='*60}")
            print(f"Diagnosing {cache}")
            print('='*60)
            diagnose_lmdb(cache)
        else:
            print(f"Cache not found: {cache}")