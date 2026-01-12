import os
from copy import deepcopy

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from unichiflow.data.pep_dataloader import PepDataset, PaddingCollate
from unichiflow.data import utils as du
from unichiflow.data.writers import save_pdb
from unichiflow.data.pep_constants import BBHeavyAtom
from unichiflow.models.chiralformer import UniChiFlowModel
 


config_path = os.path.join(os.path.dirname(__file__), "..", "configs")
config_name = "unichiflow_pepmerge.yaml"


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    du.seed_all(int(cfg.train.seed))

    sam = cfg.sample
    device = str(sam.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

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

    ds = PepDataset(
        structure_dir=str(cfg.data.structure_dir),
        dataset_dir=str(cfg.data.cache_dir),
        name="pepmerge_test",
        x_mirror=False,
        reset=bool(cfg.data.reset_cache),
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
            print("Adapting backbone weights for LoRA model structure...")
            new_state_dict = {}
            for k, v in state_dict.items():
                # Check if this key corresponds to a module targeted by LoRA
                # Standard targets are ["attn_out", "ffn"] which correspond to:
                # - self_attn.out_proj
                # - linear1, linear2
                # We check if the key belongs to these modules and if it is a weight/bias
                
                # Heuristic: if the key ends in weight/bias and the model expects .base.weight/bias
                # However, simpler is to try to match based on the error message pattern.
                # 'trunk.layers.0.linear1.weight' -> 'trunk.layers.0.linear1.base.weight'
                
                # We can iterate over model named parameters to find the correct mapping, 
                # or just use string replacement for known LoRA targets.
                # Targets defined in config or default: attn_out, ffn
                
                is_lora_target = False
                # Checking common patterns for the LoRA targets 'attn_out' and 'ffn'
                if ".self_attn.out_proj." in k or ".linear1." in k or ".linear2." in k:
                    if k.endswith(".weight"):
                        k_new = k.replace(".weight", ".base.weight")
                        new_state_dict[k_new] = v
                        is_lora_target = True
                    elif k.endswith(".bias"):
                        k_new = k.replace(".bias", ".base.bias")
                        new_state_dict[k_new] = v
                        is_lora_target = True
                
                if not is_lora_target:
                    new_state_dict[k] = v
            state_dict = new_state_dict

        # Use strict=False to allow flexibility (e.g. loading backbone into LoRA model)
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

    for idx in tqdm(range(min(len(ds), 50)), desc="inference"):
        data_list = [deepcopy(ds[idx]) for _ in range(int(sam.num_samples))]
        batch = collate(data_list)
        # overwrite chirality (0=L,1=D) for peptide decoding
        batch['chirality'] = torch.full((len(data_list),), int(sam.chirality_token), dtype=torch.long)

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

        # Use predicted peptide, keep receptor from GT
        pos_out = torch.where(batch['generate_mask'][:, :, None, None], pos15_pred, batch['pos_heavyatom'])
        mask_out = torch.where(batch['generate_mask'][:, :, None], mask15_pred, batch['mask_heavyatom'])

        # If we generated D-peptides (chirality=1), the dataset provided Mirrored Receptor (D) and Model predicted Mirrored (L-like) Peptide.
        # So pos_out is currently in the Mirror Universe (D-Rec + L-Pep).
        # We want to save the Real Universe structure (L-Rec + D-Pep).
        # So we flip everything back: pos -> -pos.
        if int(sam.chirality_token) == 1:
            pos_out = -pos_out

        name = batch['id'][0]
        save_dir = os.path.join(out_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        # Fix collate chain_id/icode back to lists
        chain_id = [list(item) for item in zip(*batch['chain_id'])][0]
        icode = [list(item) for item in zip(*batch['icode'])][0]

        for j in range(pos_out.shape[0]):
            data_saved = {
                'chain_nb': batch['chain_nb'][0].detach().cpu(),
                'chain_id': chain_id,
                'resseq': batch['resseq'][0].detach().cpu(),
                'icode': icode,
                'aa': sample['seq'][j].detach().cpu(),
                'mask_heavyatom': mask_out[j].detach().cpu(),
                'pos_heavyatom': pos_out[j].detach().cpu(),
            }
            save_pdb(data_saved, path=os.path.join(save_dir, f"sample_{j}.pdb"))

        gt_data = {
            'chain_nb': batch['chain_nb'][0].detach().cpu(),
            'chain_id': chain_id,
            'resseq': batch['resseq'][0].detach().cpu(),
            'icode': icode,
            'aa': batch['aa'][0].detach().cpu(),
            'mask_heavyatom': batch['mask_heavyatom'][0].detach().cpu(),
            'pos_heavyatom': batch['pos_heavyatom'][0].detach().cpu(),
        }
        save_pdb(gt_data, path=os.path.join(save_dir, "gt.pdb"))


if __name__ == "__main__":
    main()
