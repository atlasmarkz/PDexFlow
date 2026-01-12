import os
from copy import deepcopy
from typing import Dict

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from unichiflow.data.pep_dataloader import PepDataset, PaddingCollate
from unichiflow.data import utils as du
from unichiflow.data.pep_constants import BBHeavyAtom
from unichiflow.data.writers import save_pdb
from unichiflow.metrics import aar_accuracy, chirality_error, kabsch_rmsd
from unichiflow.models.chiralformer import UniChiFlowModel


config_path = os.path.join(os.path.dirname(__file__), "..", "configs")
config_name = "unichiflow_pepmerge.yaml"


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


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    du.seed_all(int(cfg.train.seed))

    device = str(cfg.sample.device if "sample" in cfg else cfg.train.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # dataset
    ds = PepDataset(
        structure_dir=str(cfg.data.structure_dir),
        dataset_dir=str(cfg.data.cache_dir),
        name="pepmerge_test",
        x_mirror=False,
        reset=bool(cfg.data.reset_cache),
    )
    collate = PaddingCollate(eight=False)

    # model
    model = UniChiFlowModel(
        d_model=int(cfg.model.d_model),
        n_layers=int(cfg.model.n_layers),
        n_heads=int(cfg.model.n_heads),
        dropout=float(cfg.model.dropout),
        max_len=int(cfg.model.max_len),
        aatype_vocab=int(cfg.model.aatype_vocab),
        sigma_trans=float(cfg.flow.sigma_trans),
        chi_lora=cfg.get("chi_lora"),
        esm=cfg.get("esm"),
        use_ot=bool(cfg.flow.get("use_ot", True)),
    ).to(device)

    sam = cfg.sample
    if sam.ckpt_path:
        ckpt = torch.load(str(sam.ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

    model.eval()

    max_eval = int(sam.get("max_eval", 32))
    num_steps = int(sam.get("num_steps", cfg.flow.get("num_steps", 30)))
    hard_decoder = bool(sam.get("use_hard_decoder", True))

    motif_accs = []
    chiral_errs = []
    rmsds = []

    pbar = tqdm(range(min(len(ds), max_eval)), desc="eval")
    for idx in pbar:
        data_list = [deepcopy(ds[idx])]
        batch = collate(data_list)
        batch['chirality'] = batch.get('chirality', batch.get('chirality_token', torch.zeros((len(data_list),), dtype=torch.long)))
        batch = du.recursive_to(batch, device)

        with torch.no_grad():
            sample = model.sample(batch, num_steps=num_steps, angle_purify=bool(cfg.flow.angle_purify))
            pos15_pred, mask15_pred = model.decode_to_heavyatom15(
                aa=sample['seq'],
                trans=sample['trans'],
                rotmats=sample['rotmats'],
                angles=sample['angles'],
                chirality=batch['chirality'],
                peptide_mask=batch['generate_mask'],
                hard_constraints=hard_decoder,
            )

        # masks
        pep_mask = batch['generate_mask'].bool() & batch['res_mask'].bool()
        aa_mask = pep_mask

        # motif recovery (per residue AA recovery on peptide positions)
        motif_acc = aar_accuracy(sample['seq'], batch['aa'], mask=aa_mask)
        motif_accs.append(motif_acc.detach().cpu())

        # chirality error on backbone atoms
        bb_pred = _gather_backbone(pos15_pred)
        bb_gt = _gather_backbone(batch['pos_heavyatom'])
        chiral_err = chirality_error(bb_pred['N'], bb_pred['CA'], bb_pred['C'], bb_pred['CB'], batch['chirality'])
        chiral_errs.append(chiral_err.detach().cpu())

        # scRMSD on CA (peptide positions)
        rmsd = kabsch_rmsd(_select_masked(bb_pred['CA'], aa_mask), _select_masked(bb_gt['CA'], aa_mask), mask=aa_mask)
        rmsds.append(rmsd.detach().cpu())

        pbar.set_description(
            f"motif {torch.stack(motif_accs).mean():.3f} | chiral {torch.stack(chiral_errs).mean():.3f} | rmsd {torch.stack(rmsds).mean():.3f}"
        )

    motif_mean = torch.stack(motif_accs).mean() if motif_accs else torch.tensor(0.0)
    chiral_mean = torch.stack(chiral_errs).mean() if chiral_errs else torch.tensor(0.0)
    rmsd_mean = torch.stack(rmsds).mean() if rmsds else torch.tensor(0.0)

    print("\n=== Evaluation Summary ===")
    print(f"Motif Recovery: {motif_mean.item()*100:.2f}%")
    print(f"Chiral Error: {chiral_mean.item()*100:.2f}%")
    print(f"scRMSD (CA): {rmsd_mean.item():.3f}")
    print(f"Conv. NFE (num_steps used): {num_steps}")

    # Optional: save last sample PDBs for inspection
    out_dir = str(sam.get("output_dir", "./eval_outputs"))
    os.makedirs(out_dir, exist_ok=True)
    name = batch['id'][0]
    save_dir = os.path.join(out_dir, name)
    os.makedirs(save_dir, exist_ok=True)
    # save predicted
    data_saved = {
        'chain_nb': batch['chain_nb'][0].detach().cpu(),
        'chain_id': [list(item) for item in zip(*batch['chain_id'])][0],
        'resseq': batch['resseq'][0].detach().cpu(),
        'icode': [list(item) for item in zip(*batch['icode'])][0],
        'aa': sample['seq'][0].detach().cpu(),
        'mask_heavyatom': mask15_pred[0].detach().cpu(),
        'pos_heavyatom': pos15_pred[0].detach().cpu(),
    }
    save_pdb(data_saved, path=os.path.join(save_dir, "pred_eval.pdb"))
    # save GT
    gt_data = {
        'chain_nb': batch['chain_nb'][0].detach().cpu(),
        'chain_id': [list(item) for item in zip(*batch['chain_id'])][0],
        'resseq': batch['resseq'][0].detach().cpu(),
        'icode': [list(item) for item in zip(*batch['icode'])][0],
        'aa': batch['aa'][0].detach().cpu(),
        'mask_heavyatom': batch['mask_heavyatom'][0].detach().cpu(),
        'pos_heavyatom': batch['pos_heavyatom'][0].detach().cpu(),
    }
    save_pdb(gt_data, path=os.path.join(save_dir, "gt_eval.pdb"))


if __name__ == "__main__":
    main()
