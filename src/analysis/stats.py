"""Small statistics / CSV helpers used by the evaluation entry point."""
import csv
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


def rank_array(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape or xa.size < 2:
        return float("nan")
    xr = rank_array(xa)
    yr = rank_array(ya)
    xc = xr - xr.mean()
    yc = yr - yr.mean()
    denom = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((xc * yc).sum() / denom)


def bucketize(value: int, edges: Iterable[int]) -> str:
    sorted_edges = sorted(int(e) for e in edges)
    left = 0
    for edge in sorted_edges:
        if value <= edge:
            return f"{left}-{edge}"
        left = edge + 1
    return f"{left}+"


def bucketize_float(value: float, edges: Iterable[float]) -> str:
    if np.isnan(value):
        return "nan"
    sorted_edges = sorted(float(e) for e in edges)
    left = 0.0
    for edge in sorted_edges:
        if value <= edge:
            return f"{left:.2f}-{edge:.2f}"
        left = edge
    return f"{left:.2f}+"


def mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def summarize_md_csv(md_csv_path: str) -> Dict[str, float]:
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
        "md_rmsd_last_mean": mean(rmsd_last),
        "md_rmsd_auc_mean": mean(rmsd_auc),
        "md_hbond_retention_mean": mean(hbond_ret),
    }


def summarize_orthogonal_csv(orth_csv_path: str, rosetta_col: str, orth_col: str) -> Dict[str, float]:
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
        "orth_rosetta_mean": mean(rosetta),
        "orth_score_mean": mean(orth),
        "orth_spearman": spearman_corr(rosetta, orth),
    }


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def safe_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())
