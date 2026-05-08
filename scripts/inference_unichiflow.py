"""Inference entrypoint for UniChiFlow.

Device selection goes through `resolve_inference_device` (components.py), which
auto-falls-back `cuda -> mps -> cpu` so the same yaml configs work on both a
CUDA box and a Mac.

Mac / MPS performance note (measured on Apple Silicon, 4.8M-param model,
20-step ODE sampling):

    batch  CPU     MPS     MPS/CPU
    B=2    ~100ms  ~140ms  0.7x  (MPS slower — kernel-launch overhead wins
                                  over the per-step small-op body)
    B=8    ~200ms  ~145ms  1.4x
    B=32   ~450ms  ~240ms  1.9x

So for single-sample / B<=4 interactive inference on a Mac, CPU is the faster
path — override with `sample.device=cpu` on the hydra CLI. MPS only pays off
when you batch many samples (sweeping chirality / seeds / candidates).
"""
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.pep_dataloader import PaddingCollate
from src.data.real_d_pep_dataset import RealDPepDataset
from src.data.pepmerge_dataset import PepMergeDataset
from src.data import utils as du
from src.data.writers import save_pdb
from src.data.pep_constants import BBHeavyAtom
from src.models.chiralformer import UniChiFlowModel
from src.checkpoint import load_backbone_into_lora
from src.experiments.components import resolve_inference_device
 


config_path = os.path.join(os.path.dirname(__file__), "..", "configs")
config_name = "unichiflow_inference.yaml"


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    du.seed_all(int(cfg.train.seed))

    sam = cfg.sample
    requested_device = str(sam.device)
    device = resolve_inference_device(requested_device)
    if device != requested_device:
        print(f"[device] requested={requested_device} -> resolved={device} (auto fallback)")
    else:
        print(f"[device] {device}")

    import time
    timestamp = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    
    # Base output dir from config or command line
    base_out_dir = str(sam.output_dir)
    # Ensure it saves into unichiflow_output/timestamp/
    if "unichiflow_output" not in base_out_dir:
         # Fallback if user provides a weird path, though usually they provide unichiflow_output
         base_out_dir = os.path.join(os.path.dirname(base_out_dir), "unichiflow_output")
    
    # Create the timestamped subdirectory
    out_dir = os.path.join(base_out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving inference results to: {out_dir}")

    if bool(cfg.data.get("use_pepmerge", False)):
        # Align LMDB name with existing caches: pepmerge_{split}_structure_cache.lmdb
        ds = PepMergeDataset(
            structure_dir=str(cfg.data.structure_dir),
            cache_dir=str(cfg.data.cache_dir),
            name=f"pepmerge_{cfg.data.split}",
            reset=bool(cfg.data.reset_cache),
        )
    else:
        ds = RealDPepDataset(
            structure_dir=str(cfg.data.structure_dir),
            cache_dir=str(cfg.data.cache_dir),
            name="real_d_test",
            reset=bool(cfg.data.reset_cache),
            peptide_subdir=cfg.data.get("peptide_subdir"),
            skip_receptor=bool(cfg.data.get("skip_receptor", True)),
        )

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

    if sam.ckpt_path:
        print(f"Loading backbone from {sam.ckpt_path}")
        # weights_only=False allows loading OmegaConf DictConfig stored in the checkpoint
        ckpt = torch.load(str(sam.ckpt_path), map_location=device, weights_only=False)
        state_dict = ckpt['model']

        # If LoRA is enabled, we need to adapt the backbone state dict
        # because the LoRA wrapper changes parameter names (e.g. 'linear.weight' -> 'linear.base.weight')
        if cfg.get("chi_lora", {}).get("enable", False):
            print("Adapting backbone weights for LoRA model structure (using intelligent loader)...")
            model = load_backbone_into_lora(model, state_dict)
        else:
            # Use strict=False to allow flexibility
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"Load status: {msg}")

    # Support loading separate LoRA weights if provided
    if sam.get("lora_ckpt_path") and cfg.get("chi_lora", {}).get("enable", False):
        print(f"Loading LoRA weights from {sam.lora_ckpt_path}")
        lora_ckpt = torch.load(str(sam.lora_ckpt_path), map_location=device, weights_only=False)
        msg_lora = model.load_state_dict(lora_ckpt['model'], strict=False)
        print(f"LoRA Load status: {msg_lora}")

    model.eval()

    collate = PaddingCollate(eight=False)

    for idx in tqdm(range(min(len(ds), 10)), desc="inference"):
        data_list = [deepcopy(ds[idx]) for _ in range(int(sam.num_samples))]
        batch = collate(data_list)

        # generate_mask to 1 if need peptide+receptor together
        if bool(getattr(sam, "gen_receptor", False)):
            batch['generate_mask'] = torch.ones_like(batch['aa'], dtype=torch.bool)

        # overwrite chirality (0=L,1=D) for decoding
        batch['chirality'] = torch.full((len(data_list),), int(sam.chirality_token), dtype=torch.long)

        # Coordinate scaling lives entirely inside UniChiFlowModel now:
        #   - sample() scales batch['pos_heavyatom'] on entry,
        #   - sample() unscales trans on return,
        # so batch and model outputs are consistently in raw angstroms here.
        batch = du.recursive_to(batch, device)
        sample = model.sample(batch, num_steps=int(sam.num_steps), angle_purify=bool(cfg.flow.angle_purify))

        pos15_pred, mask15_pred = model.decode_to_heavyatom15(
            aa=sample['seq'],
            trans=sample['trans'],
            rotmats=sample['rotmats'],
            angles=sample['angles'],
            chirality=batch['chirality'],
            peptide_mask=batch['generate_mask'],
            hard_constraints=bool(cfg.sample.get("use_hard_decoder", True)),
        )

        # Use predicted peptide, keep receptor from GT (both in raw angstroms)
        pos_out = torch.where(batch['generate_mask'][:, :, None, None], pos15_pred, batch['pos_heavyatom'])
        mask_out = torch.where(batch['generate_mask'][:, :, None], mask15_pred, batch['mask_heavyatom'])

        # NOTE: WE REMOVED THE GLOBAL FLIP HERE.
        # The Decoder (decode_to_heavyatom15) with hard_constraints=True ALREADY generates D-geometric coordinates
        # (via Z-flip) for D-peptides.
        # If we flip again here (Global -1), we invert the D-peptide back to L, destroying the chirality.
        # We assume the Receptor input was Real (L) and we want D-peptide output.
        # if int(sam.chirality_token) == 1:
        #     pos_out = -pos_out

        name = batch['id'][0]
        save_dir = os.path.join(out_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        # Fix collate chain_id/icode back to lists
        chain_id = [list(item) for item in zip(*batch['chain_id'])][0]
        icode = [list(item) for item in zip(*batch['icode'])][0]

        pos_out_save = pos_out
        pos_gt_save = batch['pos_heavyatom']

        for j in range(pos_out.shape[0]):
            data_saved = {
                'chain_nb': batch['chain_nb'][0].detach().cpu(),
                'chain_id': chain_id,
                'resseq': batch['resseq'][0].detach().cpu(),
                'icode': icode,
                'aa': sample['seq'][j].detach().cpu(),
                'mask_heavyatom': mask_out[j].detach().cpu(),
                'pos_heavyatom': pos_out_save[j].detach().cpu(),
            }
            save_pdb(data_saved, path=os.path.join(save_dir, f"sample_{j}.pdb"))

        gt_data = {
            'chain_nb': batch['chain_nb'][0].detach().cpu(),
            'chain_id': chain_id,
            'resseq': batch['resseq'][0].detach().cpu(),
            'icode': icode,
            'aa': batch['aa'][0].detach().cpu(),
            'mask_heavyatom': batch['mask_heavyatom'][0].detach().cpu(),
            'pos_heavyatom': pos_gt_save[0].detach().cpu(),
        }
        save_pdb(gt_data, path=os.path.join(save_dir, "gt.pdb"))


if __name__ == "__main__":
    main()
