import json
import os
import time
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple


import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.data import utils as du
from src.data.pep_constants import BBHeavyAtom
from src.data.pep_dataloader import PaddingCollate, PepDataset
from src.data.writers import save_pdb
from src.analysis.novelty import descriptor_from_ca, greedy_cluster, load_reference_bank, novelty_against_refs, pca2
from src.analysis.stats import safe_float, bucketize, bucketize_float, mean, percentile, summarize_md_csv, summarize_orthogonal_csv, write_csv
from src.analysis.metrics import aar_accuracy, chirality_error, kabsch_rmsd
from src.training.builders import load_model_for_inference, resolve_inference_device

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))  # independent of the cwd
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


def _extract_pred_peptide_seq_and_ca(batch: Dict[str, torch.Tensor], seq_pred: torch.Tensor, pos15_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # current eval loop uses batch size == 1
    pep_mask = (batch["generate_mask"][0].bool() & batch["res_mask"][0].bool()).detach().cpu()
    seq = seq_pred[0].detach().cpu()[pep_mask]
    ca = pos15_pred[0].detach().cpu()[pep_mask, BBHeavyAtom.CA]
    return seq, ca


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
        ref_bank = load_reference_bank(ds_ref, max_samples=int(cfg.eval.reference.max_samples))
        ref_features = [descriptor_from_ca(r["ca"]) for r in ref_bank]

    model = load_model_for_inference(cfg, device)
    sam = cfg.sample

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

        aar = safe_float(aar_accuracy(sample["seq"], batch["aa"], mask=aa_mask))
        bb_pred = _gather_backbone(pos15_pred)
        chiral_mask = pep_mask & mask15_pred[..., BBHeavyAtom.CB].bool()  # generated residues with a decoded CB (no GLY / unknown tokens)
        chiral_err = safe_float(chirality_error(bb_pred["N"], bb_pred["CA"], bb_pred["C"], bb_pred["CB"], batch["chirality"], mask=chiral_mask))
        bb_gt = _gather_backbone(batch["pos_heavyatom"])
        rmsd = safe_float(kabsch_rmsd(_select_masked(bb_pred["CA"], aa_mask), _select_masked(bb_gt["CA"], aa_mask), mask=aa_mask))

        pred_seq, pred_ca = _extract_pred_peptide_seq_and_ca(batch, sample["seq"], pos15_pred)
        novelty_seq_id, novelty_rmsd = novelty_against_refs(pred_seq, pred_ca, ref_bank) if ref_bank else (float("nan"), float("nan"))
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

        feat = descriptor_from_ca(pred_ca)
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

        pbar.set_description(f"AAR {mean([r['aar'] for r in per_sample]):.3f} | chiral {mean([r['chiral_err'] for r in per_sample]):.3f} | rmsd {mean([r['rmsd_ca'] for r in per_sample]):.3f}")

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
        pep_groups[bucketize(int(r["peptide_len"]), pep_bins)].append(r)
        rec_groups[bucketize(int(r["receptor_len"]), rec_bins)].append(r)
        sim_groups[bucketize_float(float(r["seq_id_to_train_max"]), sim_edges)].append(r)

    def _group_rows(groups: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for g, items in sorted(groups.items(), key=lambda kv: kv[0]):
            rows.append(
                {
                    "bucket": g,
                    "n": len(items),
                    "aar_mean": mean([float(x["aar"]) for x in items]),
                    "chiral_err_mean": mean([float(x["chiral_err"]) for x in items]),
                    "rmsd_ca_mean": mean([float(x["rmsd_ca"]) for x in items]),
                    "latency_ms_mean": mean([float(x["latency_ms"]) for x in items]),
                    "peak_mem_mb_mean": mean([float(x["peak_mem_mb"]) for x in items if not np.isnan(float(x["peak_mem_mb"]))]),
                }
            )
        return rows

    pep_scaling_rows = _group_rows(pep_groups)
    rec_scaling_rows = _group_rows(rec_groups)
    sim_split_rows = _group_rows(sim_groups)

    # diversity + embeddings
    gen_feat_mat = torch.stack(gen_features, dim=0) if gen_features else torch.zeros((0, 20), dtype=torch.float32)
    cluster_labels, n_clusters, cluster_entropy = greedy_cluster(
        gen_feat_mat, distance_threshold=float(cfg.eval.diversity.cluster_distance_threshold)
    )
    all_features = gen_features + ref_features[: int(cfg.eval.reference.embed_max_samples)] if ref_features else gen_features
    all_labels = (["generated"] * len(gen_features)) + (["reference"] * min(len(ref_features), int(cfg.eval.reference.embed_max_samples)))
    all_keys = gen_feature_keys + [f"ref_{i}" for i in range(min(len(ref_features), int(cfg.eval.reference.embed_max_samples)))]
    embed_mat = torch.stack(all_features, dim=0) if all_features else torch.zeros((0, 20), dtype=torch.float32)
    embed_xy = pca2(embed_mat)
    embed_rows = [{"key": all_keys[i], "label": all_labels[i], "x": float(embed_xy[i, 0].item()), "y": float(embed_xy[i, 1].item())} for i in range(embed_xy.shape[0])]

    # optional MD / orthogonal summaries from external results
    dynamic_summary: Dict[str, float] = {}
    if bool(cfg.eval.dynamic.enable):
        dynamic_summary = summarize_md_csv(str(cfg.eval.dynamic.md_csv))

    orth_summary: Dict[str, float] = {}
    if bool(cfg.eval.orthogonal.enable):
        orth_summary = summarize_orthogonal_csv(
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
        "aar_mean": mean([float(r["aar"]) for r in per_sample]),
        "chiral_error_mean": mean([float(r["chiral_err"]) for r in per_sample]),
        "rmsd_ca_mean": mean([float(r["rmsd_ca"]) for r in per_sample]),
        "latency_ms_mean": mean([float(r["latency_ms"]) for r in per_sample]),
        "latency_ms_p50": percentile([float(r["latency_ms"]) for r in per_sample], 50.0),
        "latency_ms_p95": percentile([float(r["latency_ms"]) for r in per_sample], 95.0),
        "peak_mem_mb_mean": mean([float(r["peak_mem_mb"]) for r in per_sample if not np.isnan(float(r["peak_mem_mb"]))]),
        "nfe_mean": mean([float(r["nfe"]) for r in per_sample]),
        "novelty_seq_id_max_mean": mean([float(r["novelty_seq_id_max"]) for r in per_sample if not np.isnan(float(r["novelty_seq_id_max"]))]),
        "novelty_rmsd_nn_mean": mean([float(r["novelty_rmsd_nn"]) for r in per_sample if not np.isnan(float(r["novelty_rmsd_nn"]))]),
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
    write_csv(per_sample_path, per_sample, list(per_sample[0].keys()))
    write_csv(os.path.join(out_dir, "scaling_by_peptide_len.csv"), pep_scaling_rows, list(pep_scaling_rows[0].keys()) if pep_scaling_rows else ["bucket", "n"])
    write_csv(os.path.join(out_dir, "scaling_by_receptor_size.csv"), rec_scaling_rows, list(rec_scaling_rows[0].keys()) if rec_scaling_rows else ["bucket", "n"])
    write_csv(os.path.join(out_dir, "generalization_similarity_split.csv"), sim_split_rows, list(sim_split_rows[0].keys()) if sim_split_rows else ["bucket", "n"])
    write_csv(os.path.join(out_dir, "embedding_2d.csv"), embed_rows, ["key", "label", "x", "y"])

    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("\n=== Evaluation Summary ===")
    for k in sorted(summary.keys()):
        print(f"{k}: {summary[k]}")


if __name__ == "__main__":
    main()
