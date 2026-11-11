"""Novelty / diversity helpers for evaluation: sequence identity, CA descriptors, clustering, reference bank."""
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from src.analysis.metrics import kabsch_rmsd
from src.analysis.stats import safe_float
from src.data.pep_constants import BBHeavyAtom
from src.data.pep_dataloader import PepDataset


def seq_identity(seq_a: torch.Tensor, seq_b: torch.Tensor) -> float:
    la, lb = int(seq_a.numel()), int(seq_b.numel())
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    n = min(la, lb)
    denom = max(la, lb)
    matches = (seq_a[:n] == seq_b[:n]).sum().item()
    return float(matches) / float(denom)


def kabsch_rmsd_unbatched(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.shape != y.shape or x.ndim != 2 or x.shape[-1] != 3:
        return float("inf")
    if x.shape[0] == 0:
        return float("inf")
    mask = torch.ones((1, x.shape[0]), dtype=torch.bool, device=x.device)
    val = kabsch_rmsd(x.unsqueeze(0), y.unsqueeze(0), mask=mask)
    return safe_float(val)


def descriptor_from_ca(ca: torch.Tensor, num_hist_bins: int = 16, max_dist: float = 20.0) -> torch.Tensor:
    if ca.ndim != 2 or ca.shape[-1] != 3:
        raise ValueError(f"Expected CA coordinates with shape (L,3), got {tuple(ca.shape)}")
    L = ca.shape[0]
    if L == 0:
        return torch.zeros((num_hist_bins + 4,), dtype=torch.float32)
    center = ca.mean(dim=0, keepdim=True)
    rg = torch.sqrt(torch.clamp(((ca - center) ** 2).sum(dim=-1).mean(), min=1e-8))
    if L > 1:
        bonds = torch.linalg.norm(ca[1:] - ca[:-1], dim=-1)
        bond_mean = bonds.mean()
        bond_std = bonds.std(unbiased=False)
    else:
        bond_mean = torch.tensor(0.0, dtype=ca.dtype)
        bond_std = torch.tensor(0.0, dtype=ca.dtype)
    dmat = torch.cdist(ca.unsqueeze(0), ca.unsqueeze(0)).squeeze(0)
    tri = dmat[torch.triu_indices(L, L, offset=1)[0], torch.triu_indices(L, L, offset=1)[1]]
    if tri.numel() == 0:
        hist = torch.zeros((num_hist_bins,), dtype=ca.dtype)
    else:
        hist = torch.histc(torch.clamp(tri, min=0.0, max=max_dist), bins=num_hist_bins, min=0.0, max=max_dist)
        hist = hist / torch.clamp(hist.sum(), min=1.0)
    return torch.cat([torch.tensor([float(L), float(rg), float(bond_mean), float(bond_std)], dtype=torch.float32), hist.float()], dim=0)


def greedy_cluster(features: torch.Tensor, distance_threshold: float) -> Tuple[List[int], int, float]:
    if features.ndim != 2:
        raise ValueError(f"Expected feature matrix (N,D), got {tuple(features.shape)}")
    n = int(features.shape[0])
    if n == 0:
        return [], 0, float("nan")
    centers: List[torch.Tensor] = []
    labels: List[int] = []
    counts: List[int] = []
    for i in range(n):
        x = features[i]
        if not centers:
            centers.append(x)
            labels.append(0)
            counts.append(1)
            continue
        dists = torch.stack([torch.linalg.norm(x - c) for c in centers])
        min_dist, min_idx = torch.min(dists, dim=0)
        if float(min_dist.item()) <= float(distance_threshold):
            cid = int(min_idx.item())
            labels.append(cid)
            counts[cid] += 1
            centers[cid] = centers[cid] + (x - centers[cid]) / float(counts[cid])
        else:
            centers.append(x)
            labels.append(len(centers) - 1)
            counts.append(1)
    probs = torch.tensor(counts, dtype=torch.float32)
    probs = probs / torch.clamp(probs.sum(), min=1.0)
    entropy = float((-probs * torch.log(torch.clamp(probs, min=1e-8))).sum().item())
    return labels, len(centers), entropy


def pca2(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected feature matrix (N,D), got {tuple(features.shape)}")
    if features.shape[0] == 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    x = features.float()
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] == 1:
        return torch.zeros((1, 2), dtype=torch.float32)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    comps = vh[:2].T
    if comps.shape[1] < 2:
        comps = torch.nn.functional.pad(comps, (0, 2 - comps.shape[1]), mode="constant", value=0.0)
    return x @ comps[:, :2]


def load_reference_bank(ds: PepDataset, max_samples: int) -> List[Dict[str, torch.Tensor]]:
    n = min(len(ds), int(max_samples))
    refs: List[Dict[str, torch.Tensor]] = []
    for i in tqdm(range(n), desc="load_ref_bank", leave=False):
        data = ds[i]
        data = dict(data)
        data["res_mask"] = torch.ones_like(data["aa"], dtype=torch.bool)
        seq, ca = extract_peptide_seq_and_ca(data)
        refs.append({"seq": seq, "ca": ca, "len": torch.tensor(int(seq.numel()))})
    return refs


def novelty_against_refs(pred_seq: torch.Tensor, pred_ca: torch.Tensor, refs: List[Dict[str, torch.Tensor]]) -> Tuple[float, float]:
    if not refs:
        return float("nan"), float("nan")
    best_seq_id = 0.0
    best_rmsd = float("inf")
    L = int(pred_seq.numel())
    for r in refs:
        seq_id = seq_identity(pred_seq, r["seq"])
        if seq_id > best_seq_id:
            best_seq_id = seq_id
        if int(r["len"].item()) == L and L >= 3:
            rmsd = kabsch_rmsd_unbatched(pred_ca, r["ca"])
            if rmsd < best_rmsd:
                best_rmsd = rmsd
    if best_rmsd == float("inf"):
        best_rmsd = float("nan")
    return best_seq_id, best_rmsd


def extract_peptide_seq_and_ca(sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    res_mask = sample["res_mask"].bool()
    pep_mask = sample["generate_mask"].bool() & res_mask
    seq = sample["aa"][pep_mask]
    ca = sample["pos_heavyatom"][pep_mask, BBHeavyAtom.CA]
    return seq.detach().cpu(), ca.detach().cpu()
