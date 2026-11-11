"""Known-answer tests for the evaluation helpers (novelty, clustering, statistics)."""
import math

import torch

from src.analysis.novelty import descriptor_from_ca, greedy_cluster, kabsch_rmsd_unbatched, novelty_against_refs, seq_identity
from src.analysis.stats import bucketize, mean, percentile, spearman_corr


def test_seq_identity():
    a = torch.tensor([1, 2, 3, 4])
    assert seq_identity(a, a) == 1.0
    assert seq_identity(a, torch.tensor([1, 2, 9, 9])) == 0.5
    assert seq_identity(a, torch.tensor([1, 2])) == 0.5      # divided by the longer length
    assert seq_identity(torch.tensor([]), a) == 0.0


def test_kabsch_rmsd_is_invariant_to_rigid_motion():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(7, 3, generator=g)
    th = 0.8
    R = torch.tensor([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
    y = x @ R.T + torch.tensor([3.0, -1.0, 2.0])
    assert kabsch_rmsd_unbatched(x, y) < 1e-4
    assert kabsch_rmsd_unbatched(x, x + 0.5 * torch.randn(7, 3, generator=g)) > 0.1


def test_novelty_against_refs_finds_the_nearest_reference():
    g = torch.Generator().manual_seed(1)
    ca = torch.randn(5, 3, generator=g)
    refs = [{"seq": torch.tensor([0, 1, 2, 3, 4]), "ca": ca.clone(), "len": torch.tensor(5)},
            {"seq": torch.tensor([9, 9, 9, 9, 9]), "ca": torch.randn(5, 3, generator=g), "len": torch.tensor(5)}]
    seq_id, rmsd = novelty_against_refs(torch.tensor([0, 1, 2, 7, 7]), ca, refs)
    assert seq_id == 0.6 and rmsd < 1e-4
    assert all(math.isnan(v) for v in novelty_against_refs(torch.tensor([1, 2, 3]), ca, []))


def test_descriptor_and_clustering():
    g = torch.Generator().manual_seed(2)
    a = torch.randn(6, 3, generator=g) * 3
    feats = torch.stack([descriptor_from_ca(a), descriptor_from_ca(a + 5.0), descriptor_from_ca(a * 3.0)])
    assert feats.shape[0] == 3 and torch.isfinite(feats).all()
    assert torch.allclose(feats[0], feats[1], atol=1e-5), "descriptor must be translation invariant"
    labels, n_clusters, _ = greedy_cluster(feats, distance_threshold=1e-3)
    assert labels[0] == labels[1] and labels[0] != labels[2] and n_clusters == 2


def test_statistics():
    assert abs(spearman_corr([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(spearman_corr([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    assert math.isnan(spearman_corr([1], [1]))
    assert mean([1.0, 2.0, 3.0]) == 2.0 and math.isnan(mean([]))
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0
    assert bucketize(3, [5, 10]) == "0-5" and bucketize(7, [5, 10]) == "6-10" and bucketize(11, [5, 10]) == "11+"
