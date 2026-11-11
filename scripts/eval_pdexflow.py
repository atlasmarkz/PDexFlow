import csv
import json
import os
import sys
import time
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.checkpoint import load_backbone_into_lora
from src.data import utils as du
from src.data.pep_constants import BBHeavyAtom
from src.data.pep_dataloader import PaddingCollate, PepDataset
from src.data.writers import save_pdb
from src.analysis.metrics import aar_accuracy, chirality_error, kabsch_rmsd
from src.experiments.components import build_model, resolve_inference_device

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))  # independent of the cwd
config_name = "pdexflow_eval.yaml"


def _device_sync(device: str):
    """Barrier + wait-for-kernels across cuda/mps/cpu. No-op on cpu."""
    if device.startswith("cuda"):
        torch.cuda.synchronize(device=torch.device(device))
    elif device == "mps":
        # torch.mps.synchronize exists since ~2.1; guard for older wheels.
        sync = getattr(getattr(torch, "mps", None), "synchronize", None)
        if sync is not None:
            sync()


def _device_reset_peak_mem(device: str):
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device=torch.device(device))
    # mps has no peak-reset API; per-step current_allocated_memory still works.


def _device_peak_mem_mb(device: str) -> float:
    if device.startswith("cuda"):
        return float(torch.cuda.max_memory_allocated(device=torch.device(device))) / (1024.0 * 1024.0)
    if device == "mps":
        cur = getattr(getattr(torch, "mps", None), "current_allocated_memory", None)
        if cur is not None:
            # MPS has no "peak since reset"; this is instantaneous usage at
            # read time, which for a blocking single-sample eval is a
            # reasonable proxy.
            return float(cur()) / (1024.0 * 1024.0)
    return float("nan")


def _select_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    return x * mask


def _gather_backbone(pos15: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "N": pos15[:, :, BBHeavyAtom.N],
        "CA": pos15[:, :, BBHeavyAtom.CA],
        "C": pos15[:, :, BBHeavyAtom.C],
        "CB": pos15[:, :, BBHeavyAtom.CB],
    }


def _safe_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def _extract_peptide_seq_and_ca(sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    res_mask = sample["res_mask"].bool()
    pep_mask = sample["generate_mask"].bool() & res_mask
    seq = sample["aa"][pep_mask]
    ca = sample["pos_heavyatom"][pep_mask, BBHeavyAtom.CA]
    return seq.detach().cpu(), ca.detach().cpu()


def _extract_pred_peptide_seq_and_ca(batch: Dict[str, torch.Tensor], seq_pred: torch.Tensor, pos15_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # current eval loop uses batch size == 1
    pep_mask = (batch["generate_mask"][0].bool() & batch["res_mask"][0].bool()).detach().cpu()
    seq = seq_pred[0].detach().cpu()[pep_mask]
    ca = pos15_pred[0].detach().cpu()[pep_mask, BBHeavyAtom.CA]
    return seq, ca


def _seq_identity(seq_a: torch.Tensor, seq_b: torch.Tensor) -> float:
    la, lb = int(seq_a.numel()), int(seq_b.numel())
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    n = min(la, lb)
    denom = max(la, lb)
    matches = (seq_a[:n] == seq_b[:n]).sum().item()
    return float(matches) / float(denom)


def _kabsch_rmsd_unbatched(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.shape != y.shape or x.ndim != 2 or x.shape[-1] != 3:
        return float("inf")
    if x.shape[0] == 0:
        return float("inf")
    mask = torch.ones((1, x.shape[0]), dtype=torch.bool, device=x.device)
    val = kabsch_rmsd(x.unsqueeze(0), y.unsqueeze(0), mask=mask)
    return _safe_float(val)


def _descriptor_from_ca(ca: torch.Tensor, num_hist_bins: int = 16, max_dist: float = 20.0) -> torch.Tensor:
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


def _rank_array(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape or xa.size < 2:
        return float("nan")
    xr = _rank_array(xa)
    yr = _rank_array(ya)
    xc = xr - xr.mean()
    yc = yr - yr.mean()
    denom = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((xc * yc).sum() / denom)


def _bucketize(value: int, edges: Iterable[int]) -> str:
    sorted_edges = sorted(int(e) for e in edges)
    left = 0
    for edge in sorted_edges:
        if value <= edge:
            return f"{left}-{edge}"
        left = edge + 1
    return f"{left}+"


def _bucketize_float(value: float, edges: Iterable[float]) -> str:
    if np.isnan(value):
        return "nan"
    sorted_edges = sorted(float(e) for e in edges)
    left = 0.0
    for edge in sorted_edges:
        if value <= edge:
            return f"{left:.2f}-{edge:.2f}"
        left = edge
    return f"{left:.2f}+"


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _greedy_cluster(features: torch.Tensor, distance_threshold: float) -> Tuple[List[int], int, float]:
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


def _pca2(features: torch.Tensor) -> torch.Tensor:
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


def _load_reference_bank(ds: PepDataset, max_samples: int) -> List[Dict[str, torch.Tensor]]:
    n = min(len(ds), int(max_samples))
    refs: List[Dict[str, torch.Tensor]] = []
    for i in tqdm(range(n), desc="load_ref_bank", leave=False):
        data = ds[i]
        data = dict(data)
        data["res_mask"] = torch.ones_like(data["aa"], dtype=torch.bool)
        seq, ca = _extract_peptide_seq_and_ca(data)
        refs.append({"seq": seq, "ca": ca, "len": torch.tensor(int(seq.numel()))})
    return refs


def _novelty_against_refs(pred_seq: torch.Tensor, pred_ca: torch.Tensor, refs: List[Dict[str, torch.Tensor]]) -> Tuple[float, float]:
    if not refs:
        return float("nan"), float("nan")
    best_seq_id = 0.0
    best_rmsd = float("inf")
    L = int(pred_seq.numel())
    for r in refs:
        seq_id = _seq_identity(pred_seq, r["seq"])
        if seq_id > best_seq_id:
            best_seq_id = seq_id
        if int(r["len"].item()) == L and L >= 3:
            rmsd = _kabsch_rmsd_unbatched(pred_ca, r["ca"])
            if rmsd < best_rmsd:
                best_rmsd = rmsd
    if best_rmsd == float("inf"):
        best_rmsd = float("nan")
    return best_seq_id, best_rmsd


def _summarize_md_csv(md_csv_path: str) -> Dict[str, float]:
    with open(md_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"MD CSV has no rows: {md_csv_path}")
    required_cols = {"sample_id", "time_ps", "rmsd", "hbond_count"}
    header = set(rows[0].keys())
    missing = required_cols - header
    if missing:
        raise ValueError(f"MD CSV missing required columns: {sorted(missing)}")
    by_sample: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)
    rmsd_last = []
    rmsd_auc = []
    hbond_ret = []
    for _, series in by_sample.items():
        series = sorted(series, key=lambda r: float(r["time_ps"]))
        t = np.asarray([float(r["time_ps"]) for r in series], dtype=np.float64)
        r = np.asarray([float(r["rmsd"]) for r in series], dtype=np.float64)
        h = np.asarray([float(r["hbond_count"]) for r in series], dtype=np.float64)
        rmsd_last.append(float(r[-1]))
        tspan = max(float(t[-1] - t[0]), 1e-8)
        rmsd_auc.append(float(np.trapz(r, t) / tspan))
        h0 = max(float(h[0]), 1e-8)
        hbond_ret.append(float(h[-1] / h0))
    return {
        "md_n_samples": float(len(by_sample)),
        "md_rmsd_last_mean": _mean(rmsd_last),
        "md_rmsd_auc_mean": _mean(rmsd_auc),
        "md_hbond_retention_mean": _mean(hbond_ret),
    }


def _summarize_orthogonal_csv(orth_csv_path: str, rosetta_col: str, orth_col: str) -> Dict[str, float]:
    with open(orth_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Orthogonal CSV has no rows: {orth_csv_path}")
    header = set(rows[0].keys())
    for col in (rosetta_col, orth_col):
        if col not in header:
            raise ValueError(f"Orthogonal CSV missing required column: {col}")
    rosetta = [float(r[rosetta_col]) for r in rows]
    orth = [float(r[orth_col]) for r in rows]
    return {
        "orth_n_samples": float(len(rows)),
        "orth_rosetta_mean": _mean(rosetta),
        "orth_score_mean": _mean(orth),
        "orth_spearman": _spearman_corr(rosetta, orth),
    }


def _write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    du.seed_all(int(cfg.train.seed))

    requested_device = str(cfg.sample.device if "sample" in cfg else cfg.train.device)
    device = resolve_inference_device(requested_device)
    if device != requested_device:
        print(f"[device] requested={requested_device} -> resolved={device} (auto fallback)")
    else:
        print(f"[device] {device}")

    ds_eval = PepDataset(
        structure_dir=str(cfg.data.structure_dir),
        dataset_dir=str(cfg.data.cache_dir),
        name=str(cfg.eval.dataset_name),
        x_mirror=bool(cfg.eval.get("dataset_x_mirror", False)),
        reset=bool(cfg.data.reset_cache),
    )
    collate = PaddingCollate(eight=False)

    ref_bank: List[Dict[str, torch.Tensor]] = []
    ref_features: List[torch.Tensor] = []
    if bool(cfg.eval.reference.enable):
        ds_ref = PepDataset(
            structure_dir=str(cfg.data.structure_dir),
            dataset_dir=str(cfg.data.cache_dir),
            name=str(cfg.eval.reference.dataset_name),
            x_mirror=bool(cfg.eval.reference.get("x_mirror", False)),
            reset=False,
        )
        ref_bank = _load_reference_bank(ds_ref, max_samples=int(cfg.eval.reference.max_samples))
        ref_features = [_descriptor_from_ca(r["ca"]) for r in ref_bank]

    model = build_model(cfg, include_multitask_weights=False).to(device)  # same construction path as training

    sam = cfg.sample
    if not sam.ckpt_path:
        raise ValueError("sample.ckpt_path must be provided for evaluation")
    ckpt = torch.load(str(sam.ckpt_path), map_location=device, weights_only=False)
    if cfg.get("chi_lora", {}).get("enable", False):
        model = load_backbone_into_lora(model, ckpt["model"])
    else:
        try:
            model.load_state_dict(ckpt["model"], strict=True)
        except RuntimeError:
            model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    max_eval = min(len(ds_eval), int(cfg.sample.max_eval))
    num_steps = int(sam.get("num_steps", cfg.flow.get("num_steps", 30)))
    hard_decoder = bool(sam.get("use_hard_decoder", True))
    chirality_token = int(sam.get("chirality_token", 1 if bool(cfg.eval.get("dataset_x_mirror", False)) else 0))
    out_dir = str(cfg.sample.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    per_sample: List[Dict[str, object]] = []
    gen_features: List[torch.Tensor] = []
    gen_feature_keys: List[str] = []

    pbar = tqdm(range(max_eval), desc="eval")
    for idx in pbar:
        data_list = [deepcopy(ds_eval[idx])]
        batch = collate(data_list)
        # Chirality condition: explicit `sample.chirality_token`, else D for the parity-mirrored split and L otherwise.
        batch["chirality"] = torch.full((len(data_list),), chirality_token, dtype=torch.long)
        batch = du.recursive_to(batch, device)

        _device_reset_peak_mem(device)
        _device_sync(device)
        tic = time.perf_counter()

        with torch.no_grad():
            sample = model.sample(batch, num_steps=num_steps, angle_purify=bool(cfg.flow.angle_purify))
            pos15_pred, mask15_pred = model.decode_to_heavyatom15(
                aa=sample["seq"],
                trans=sample["trans"],
                rotmats=sample["rotmats"],
                angles=sample["angles"],
                chirality=batch["chirality"],
                peptide_mask=batch["generate_mask"],
                hard_constraints=hard_decoder,
            )

        _device_sync(device)
        latency_ms = (time.perf_counter() - tic) * 1000.0
        peak_mem_mb = _device_peak_mem_mb(device)

        pep_mask = batch["generate_mask"].bool() & batch["res_mask"].bool()
        aa_mask = pep_mask

        aar = _safe_float(aar_accuracy(sample["seq"], batch["aa"], mask=aa_mask))
        bb_pred = _gather_backbone(pos15_pred)
        chiral_mask = pep_mask & mask15_pred[..., BBHeavyAtom.CB].bool()  # generated residues with a decoded CB (no GLY / unknown tokens)
        chiral_err = _safe_float(chirality_error(bb_pred["N"], bb_pred["CA"], bb_pred["C"], bb_pred["CB"], batch["chirality"], mask=chiral_mask))
        bb_gt = _gather_backbone(batch["pos_heavyatom"])
        rmsd = _safe_float(kabsch_rmsd(_select_masked(bb_pred["CA"], aa_mask), _select_masked(bb_gt["CA"], aa_mask), mask=aa_mask))

        pred_seq, pred_ca = _extract_pred_peptide_seq_and_ca(batch, sample["seq"], pos15_pred)
        novelty_seq_id, novelty_rmsd = _novelty_against_refs(pred_seq, pred_ca, ref_bank) if ref_bank else (float("nan"), float("nan"))
        seq_id_to_train = novelty_seq_id

        peptide_len = int((batch["generate_mask"][0].bool() & batch["res_mask"][0].bool()).sum().item())
        receptor_len = int(((~batch["generate_mask"][0].bool()) & batch["res_mask"][0].bool()).sum().item())
        sample_id = str(batch["id"][0])
        key = f"{sample_id}__{idx}"

        record = {
            "key": key,
            "sample_id": sample_id,
            "idx": idx,
            "peptide_len": peptide_len,
            "receptor_len": receptor_len,
            "aar": aar,
            "chiral_err": chiral_err,
            "rmsd_ca": rmsd,
            "latency_ms": latency_ms,
            "peak_mem_mb": peak_mem_mb,
            "nfe": num_steps,
            "novelty_seq_id_max": novelty_seq_id,
            "novelty_rmsd_nn": novelty_rmsd,
            "seq_id_to_train_max": seq_id_to_train,
        }
        per_sample.append(record)

        feat = _descriptor_from_ca(pred_ca)
        gen_features.append(feat)
        gen_feature_keys.append(key)

        if idx < int(cfg.eval.save_examples):
            save_dir = os.path.join(out_dir, sample_id)
            os.makedirs(save_dir, exist_ok=True)
            chain_id = [list(item) for item in zip(*batch["chain_id"])][0]
            icode = [list(item) for item in zip(*batch["icode"])][0]
            pred_data = {
                "chain_nb": batch["chain_nb"][0].detach().cpu(),
                "chain_id": chain_id,
                "resseq": batch["resseq"][0].detach().cpu(),
                "icode": icode,
                "aa": sample["seq"][0].detach().cpu(),
                "mask_heavyatom": mask15_pred[0].detach().cpu(),
                "pos_heavyatom": pos15_pred[0].detach().cpu(),
            }
            gt_data = {
                "chain_nb": batch["chain_nb"][0].detach().cpu(),
                "chain_id": chain_id,
                "resseq": batch["resseq"][0].detach().cpu(),
                "icode": icode,
                "aa": batch["aa"][0].detach().cpu(),
                "mask_heavyatom": batch["mask_heavyatom"][0].detach().cpu(),
                "pos_heavyatom": batch["pos_heavyatom"][0].detach().cpu(),
            }
            save_pdb(pred_data, path=os.path.join(save_dir, "pred_eval.pdb"))
            save_pdb(gt_data, path=os.path.join(save_dir, "gt_eval.pdb"))

        pbar.set_description(f"AAR {_mean([r['aar'] for r in per_sample]):.3f} | chiral {_mean([r['chiral_err'] for r in per_sample]):.3f} | rmsd {_mean([r['rmsd_ca'] for r in per_sample]):.3f}")

    if not per_sample:
        raise RuntimeError("No samples were evaluated. Check dataset and max_eval settings.")

    # scaling summaries
    pep_bins = list(cfg.eval.scaling.peptide_bins)
    rec_bins = list(cfg.eval.scaling.receptor_bins)
    pep_groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    rec_groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    sim_groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    sim_edges = list(cfg.eval.generalization.seq_identity_bins)

    for r in per_sample:
        pep_groups[_bucketize(int(r["peptide_len"]), pep_bins)].append(r)
        rec_groups[_bucketize(int(r["receptor_len"]), rec_bins)].append(r)
        sim_groups[_bucketize_float(float(r["seq_id_to_train_max"]), sim_edges)].append(r)

    def _group_rows(groups: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for g, items in sorted(groups.items(), key=lambda kv: kv[0]):
            rows.append(
                {
                    "bucket": g,
                    "n": len(items),
                    "aar_mean": _mean([float(x["aar"]) for x in items]),
                    "chiral_err_mean": _mean([float(x["chiral_err"]) for x in items]),
                    "rmsd_ca_mean": _mean([float(x["rmsd_ca"]) for x in items]),
                    "latency_ms_mean": _mean([float(x["latency_ms"]) for x in items]),
                    "peak_mem_mb_mean": _mean([float(x["peak_mem_mb"]) for x in items if not np.isnan(float(x["peak_mem_mb"]))]),
                }
            )
        return rows

    pep_scaling_rows = _group_rows(pep_groups)
    rec_scaling_rows = _group_rows(rec_groups)
    sim_split_rows = _group_rows(sim_groups)

    # diversity + embeddings
    gen_feat_mat = torch.stack(gen_features, dim=0) if gen_features else torch.zeros((0, 20), dtype=torch.float32)
    cluster_labels, n_clusters, cluster_entropy = _greedy_cluster(
        gen_feat_mat, distance_threshold=float(cfg.eval.diversity.cluster_distance_threshold)
    )
    all_features = gen_features + ref_features[: int(cfg.eval.reference.embed_max_samples)] if ref_features else gen_features
    all_labels = (["generated"] * len(gen_features)) + (["reference"] * min(len(ref_features), int(cfg.eval.reference.embed_max_samples)))
    all_keys = gen_feature_keys + [f"ref_{i}" for i in range(min(len(ref_features), int(cfg.eval.reference.embed_max_samples)))]
    embed_mat = torch.stack(all_features, dim=0) if all_features else torch.zeros((0, 20), dtype=torch.float32)
    embed_xy = _pca2(embed_mat)
    embed_rows = [{"key": all_keys[i], "label": all_labels[i], "x": float(embed_xy[i, 0].item()), "y": float(embed_xy[i, 1].item())} for i in range(embed_xy.shape[0])]

    # optional MD / orthogonal summaries from external results
    dynamic_summary: Dict[str, float] = {}
    if bool(cfg.eval.dynamic.enable):
        dynamic_summary = _summarize_md_csv(str(cfg.eval.dynamic.md_csv))

    orth_summary: Dict[str, float] = {}
    if bool(cfg.eval.orthogonal.enable):
        orth_summary = _summarize_orthogonal_csv(
            orth_csv_path=str(cfg.eval.orthogonal.csv_path),
            rosetta_col=str(cfg.eval.orthogonal.rosetta_col),
            orth_col=str(cfg.eval.orthogonal.orth_col),
        )

    cluster_max_fraction = float("nan")
    if cluster_labels:
        counts = np.bincount(np.asarray(cluster_labels, dtype=np.int64))
        cluster_max_fraction = float(counts.max() / max(int(counts.sum()), 1))

    summary = {
        "n_eval": len(per_sample),
        "aar_mean": _mean([float(r["aar"]) for r in per_sample]),
        "chiral_error_mean": _mean([float(r["chiral_err"]) for r in per_sample]),
        "rmsd_ca_mean": _mean([float(r["rmsd_ca"]) for r in per_sample]),
        "latency_ms_mean": _mean([float(r["latency_ms"]) for r in per_sample]),
        "latency_ms_p50": _percentile([float(r["latency_ms"]) for r in per_sample], 50.0),
        "latency_ms_p95": _percentile([float(r["latency_ms"]) for r in per_sample], 95.0),
        "peak_mem_mb_mean": _mean([float(r["peak_mem_mb"]) for r in per_sample if not np.isnan(float(r["peak_mem_mb"]))]),
        "nfe_mean": _mean([float(r["nfe"]) for r in per_sample]),
        "novelty_seq_id_max_mean": _mean([float(r["novelty_seq_id_max"]) for r in per_sample if not np.isnan(float(r["novelty_seq_id_max"]))]),
        "novelty_rmsd_nn_mean": _mean([float(r["novelty_rmsd_nn"]) for r in per_sample if not np.isnan(float(r["novelty_rmsd_nn"]))]),
        "cluster_count": float(n_clusters),
        "cluster_entropy": cluster_entropy,
        "cluster_max_fraction": cluster_max_fraction,
        "training_cost_gpu_count": float(cfg.eval.training_cost.gpu_count),
        "training_cost_gpu_hours": float(cfg.eval.training_cost.gpu_hours),
        "training_cost_peak_mem_mb": float(cfg.eval.training_cost.peak_mem_mb),
    }
    summary.update(dynamic_summary)
    summary.update(orth_summary)

    # persist
    per_sample_path = os.path.join(out_dir, "per_sample_metrics.csv")
    _write_csv(per_sample_path, per_sample, list(per_sample[0].keys()))
    _write_csv(os.path.join(out_dir, "scaling_by_peptide_len.csv"), pep_scaling_rows, list(pep_scaling_rows[0].keys()) if pep_scaling_rows else ["bucket", "n"])
    _write_csv(os.path.join(out_dir, "scaling_by_receptor_size.csv"), rec_scaling_rows, list(rec_scaling_rows[0].keys()) if rec_scaling_rows else ["bucket", "n"])
    _write_csv(os.path.join(out_dir, "generalization_similarity_split.csv"), sim_split_rows, list(sim_split_rows[0].keys()) if sim_split_rows else ["bucket", "n"])
    _write_csv(os.path.join(out_dir, "embedding_2d.csv"), embed_rows, ["key", "label", "x", "y"])

    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("\n=== Evaluation Summary ===")
    for k in sorted(summary.keys()):
        print(f"{k}: {summary[k]}")


if __name__ == "__main__":
    main()
