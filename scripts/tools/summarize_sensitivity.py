import argparse
import csv
import json
from collections import defaultdict
from typing import Dict, List

import numpy as np


def _mean(xs: List[float]) -> float:
    return float(np.mean(np.asarray(xs, dtype=np.float64))) if xs else float("nan")


def _group_mean(rows: List[Dict[str, str]], group_key: str, metric_keys: List[str]) -> List[Dict[str, float]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        groups[r[group_key]].append(r)
    out = []
    for g, items in sorted(groups.items(), key=lambda kv: kv[0]):
        row: Dict[str, float] = {"group": g, "n": float(len(items))}
        for m in metric_keys:
            vals = [float(x[m]) for x in items if x[m] != ""]
            row[f"{m}_mean"] = _mean(vals)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description="Summarize robustness/hyperparameter sensitivity from experiment CSV.")
    ap.add_argument("csv_path", type=str, help="CSV with ablation runs and metrics.")
    ap.add_argument("--out_json", type=str, default="sensitivity_summary.json")
    ap.add_argument("--lambda_col", type=str, default="lambda_sym")
    ap.add_argument("--rank_col", type=str, default="lora_rank")
    ap.add_argument("--gate_col", type=str, default="gate_enabled")
    ap.add_argument("--metrics", type=str, default="chiral_err,rmsd_ca,aar", help="Comma-separated metric columns.")
    args = ap.parse_args()

    with open(args.csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {args.csv_path}")

    metric_keys = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for k in [args.lambda_col, args.rank_col, args.gate_col, *metric_keys]:
        if k not in rows[0]:
            raise ValueError(f"Missing required column: {k}")

    summary = {
        "lambda_sensitivity": _group_mean(rows, args.lambda_col, metric_keys),
        "rank_sensitivity": _group_mean(rows, args.rank_col, metric_keys),
        "gate_ablation": _group_mean(rows, args.gate_col, metric_keys),
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
