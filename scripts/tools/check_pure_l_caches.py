"""Check LMDB caches for pure-L training (TwinUniversePepDataset).

Checks both L-cache (pepmerge_train_structure_cache.lmdb) and
mirrored-cache (pepmerge_train_structure_x_cache.lmdb).
"""
import os
import pickle
import lmdb
import torch

def check_cache(lmdb_path, label="cache"):
    if not os.path.exists(lmdb_path):
        print(f"[{label}] NOT FOUND: {lmdb_path}")
        return 0, []

    try:
        env = lmdb.open(lmdb_path, readonly=True, subdir=False, lock=False, readahead=False, meminit=False)
        stats = env.stat()
        total = stats['entries']
    except Exception as e:
        print(f"[{label}] OPEN ERROR: {lmdb_path} - {e}")
        return 0, []
    finally:
        env.close()

    issues = []
    valid = 0
    env = lmdb.open(lmdb_path, readonly=True, subdir=False, lock=False, readahead=False, meminit=False)
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                sid = key.decode('utf-8')
                try:
                    data = pickle.loads(value)
                    # pos_heavyatom finite check
                    if not torch.all(torch.isfinite(data['pos_heavyatom'])):
                        issues.append(f"{sid}: pos_heavyatom NaN/Inf")
                        continue
                    # torsion_angle finite check
                    if not torch.all(torch.isfinite(data.get('torsion_angle', torch.tensor([])))):
                        issues.append(f"{sid}: torsion_angle NaN/Inf")
                        continue
                    # peptide length check
                    pep_len = data['generate_mask'].sum().item()
                    if pep_len < 3 or pep_len > 25:
                        issues.append(f"{sid}: pep_len={pep_len}")
                        continue
                    # aa range check
                    if (data['aa'] >= 20).any():
                        issues.append(f"{sid}: UNK residues")
                        continue
                    valid += 1
                except Exception as e:
                    issues.append(f"{sid}: {type(e).__name__}: {e}")
    finally:
        env.close()

    return total, issues


if __name__ == "__main__":
    base = ".cache/pep_cache"

    l_cache = os.path.join(base, "pepmerge_train_structure_cache.lmdb")
    x_cache = os.path.join(base, "pepmerge_train_structure_x_cache.lmdb")

    print("=" * 60)
    print("Checking L-peptide cache (pepmerge_train_structure_cache.lmdb)")
    print("=" * 60)
    total_l, issues_l = check_cache(l_cache, "L-cache")
    print(f"Total entries: {total_l}, Valid: {total_l - len(issues_l)}, Issues: {len(issues_l)}")
    if issues_l:
        print("\nFirst 20 issues:")
        for i in issues_l[:20]:
            print(f"  {i}")

    print("\n" + "=" * 60)
    print("Checking D/mirrored-cache (pepmerge_train_structure_x_cache.lmdb)")
    print("=" * 60)
    total_x, issues_x = check_cache(x_cache, "X-cache")
    print(f"Total entries: {total_x}, Valid: {total_x - len(issues_x)}, Issues: {len(issues_x)}")
    if issues_x:
        print("\nFirst 20 issues:")
        for i in issues_x[:20]:
            print(f"  {i}")

    print("\n" + "=" * 60)
    if not issues_l and not issues_x:
        print("RESULT: All caches OK - training can proceed")
    else:
        bad = len(issues_l) + len(issues_x)
        print(f"RESULT: {bad} corrupted entries found - rebuild with data.reset_cache=true")
    print("=" * 60)