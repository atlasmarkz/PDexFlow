"""Train / test split of PepMerge-style complexes.

The ids listed in ``test_ids.txt`` form the test split; every other complex belongs to training. The split is encoded in the
dataset name (``..._train``, ``..._test`` or ``..._all``).
"""
import os
from functools import lru_cache
from typing import Iterable, List, Set

TEST_IDS_PATH = os.path.join(os.path.dirname(__file__), "test_ids.txt")


@lru_cache(maxsize=1)
def test_ids() -> Set[str]:
    with open(TEST_IDS_PATH, "r") as f:
        return {line.strip() for line in f if line.strip()}


def filter_ids(dataset_name: str, ids: Iterable[str], strict: bool = True) -> List[str]:
    """Keep the ids that belong to the split named in ``dataset_name``.

    strict=True raises for a name without a split keyword; strict=False keeps everything in that case.
    """
    ids = list(ids)
    ids_test = test_ids()
    if "train" in dataset_name:
        return [i for i in ids if i not in ids_test]
    if "test" in dataset_name:
        return [i for i in ids if i in ids_test]
    if "all" in dataset_name or not strict:
        return ids
    raise ValueError("Data split can also accept: train/test/all.")
