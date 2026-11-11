"""List all LMDB caches and their sizes."""
import os
import lmdb

cache_dir = ".cache"
for item in sorted(os.listdir(cache_dir)):
    item_path = os.path.join(cache_dir, item)
    if not os.path.isdir(item_path):
        continue

    lmdb_files = [f for f in os.listdir(item_path) if f.endswith('.lmdb') and not f.endswith('-lock')]

    if lmdb_files:
        for lmdb_file in lmdb_files:
            lmdb_path = os.path.join(item_path, lmdb_file)
            try:
                env = lmdb.open(lmdb_path, readonly=True)
                stats = env.stat()
                print(f"{item}/{lmdb_file}: {stats['entries']} entries")
                env.close()
            except Exception as e:
                err_msg = str(e).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                print(f"{item}/{lmdb_file}: CORRUPTED - {err_msg}")
    else:
        print(f"{item}: no LMDB files")