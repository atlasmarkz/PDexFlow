"""Shared builders for PDexFlow training scripts.

Step 2 of the Trainer refactor roadmap: extract the setup-phase construction
(dataset, dataloader, model, optimizer, checkpoint load) so the two trainers
(`train_pdexflow.py`, `train_lora_real_d.py`) call into identical factories.

No behavior change. Each function reproduces the exact construction the inline
code used to perform — pre-existing inconsistencies between the two scripts
(e.g. backbone passes cfm_weight/sym_weight, LoRA does not) are preserved
here via explicit flags so they can be unified later in a dedicated pass.
"""
import os
from typing import Literal

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.checkpoint import load_backbone_into_lora
from src.data.pep_dataloader import DualPepDataset, PaddingCollate, TwinUniversePepDataset
from src.data.real_d_pep_dataset import RealDPepDataset
from src.models.chiralformer import PDexFlowModel


DatasetKind = Literal["pepmerge", "real_d"]


def build_dataset(cfg: DictConfig, kind: DatasetKind) -> Dataset:
    """Build the training dataset.

    Kind is supplied by the caller rather than inferred from cfg so the
    existing yaml configs don't need a new field. Step 4 (config schema
    unification) will revisit this.
    """
    if kind == "pepmerge":
        structure_dir_D = str(cfg.data.get("structure_dir_D", cfg.data.structure_dir))
        cache_dir_D = str(cfg.data.get("cache_dir_D", cfg.data.cache_dir))
        has_mirrored_cache = os.path.exists(os.path.join(cache_dir_D, 'pepmerge_mirrored_structure_cache.lmdb'))
        has_x_mirror_cache = os.path.exists(os.path.join(cache_dir_D, 'pepmerge_train_structure_x_cache.lmdb'))
        if structure_dir_D != str(cfg.data.structure_dir) and (has_mirrored_cache or has_x_mirror_cache):
            name_D = 'pepmerge_mirrored' if has_mirrored_cache else 'pepmerge'
            return DualPepDataset(
                structure_dir_L=str(cfg.data.structure_dir),
                structure_dir_D=structure_dir_D,
                cache_dir_L=str(cfg.data.cache_dir),
                cache_dir_D=cache_dir_D,
                name_L="pepmerge",
                name_D=name_D,
                reset=bool(cfg.data.reset_cache),
            )
        return TwinUniversePepDataset(
            structure_dir=str(cfg.data.structure_dir),
            cache_dir=str(cfg.data.cache_dir),
            split_name=f"pepmerge_{cfg.data.split}",
            reset=bool(cfg.data.reset_cache),
        )
    if kind == "real_d":
        return RealDPepDataset(
            structure_dir=str(cfg.data.structure_dir),
            cache_dir=str(cfg.data.cache_dir),
            name=f"real_d_{cfg.data.split}",
            reset=bool(cfg.data.reset_cache),
            chirality_mode=str(cfg.data.get("chirality_mode", "real_d")),
        )
    raise ValueError(f"Unknown dataset kind: {kind!r}")


def build_dataloader(ds: Dataset, cfg: DictConfig, *, eight: bool = True) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        collate_fn=PaddingCollate(eight=eight),
        pin_memory=True,
    )


def inf_iterator(loader: DataLoader):
    """Infinitely cycle a DataLoader. Shared helper — was duplicated in both scripts."""
    while True:
        for batch in loader:
            yield batch


def build_model(cfg: DictConfig, *, include_multitask_weights: bool) -> PDexFlowModel:
    """Construct PDexFlowModel from cfg.

    `include_multitask_weights` flag preserves the pre-existing asymmetry:
    the backbone trainer wires cfm_weight/sym_weight through, the LoRA
    trainer relies on the model defaults. Step 4 will unify the config
    schema; until then we faithfully reproduce both call shapes.
    """
    kwargs = dict(
        d_model=int(cfg.model.d_model),
        n_layers=int(cfg.model.n_layers),
        n_heads=int(cfg.model.n_heads),
        dropout=float(cfg.model.dropout),
        max_len=int(cfg.model.max_len),
        aatype_vocab=int(cfg.model.aatype_vocab),
        sigma_trans=float(cfg.flow.sigma_trans),
        chi_lora=cfg.get("chi_lora"),
        esm=cfg.get("esm"),
        seq_weight=float(cfg.flow.get("seq_weight", 1.0)),
        angle_dim=int(cfg.model.get("angle_dim", 5)),
        use_ot=bool(cfg.flow.get("use_ot", True)),
        coord_scale=float(cfg.model.get("coord_scale", 0.1)),
        n_rbf=int(cfg.model.get("n_rbf", 16)),
        loss_type=str(cfg.flow.get("loss_type", "huber")),
    )
    if include_multitask_weights:
        kwargs["cfm_weight"] = float(cfg.flow.get("cfm_weight", 1.0))
        kwargs["sym_weight"] = float(cfg.flow.get("sym_weight", 1.0))
    return PDexFlowModel(**kwargs)


def build_optimizer(model: PDexFlowModel, cfg: DictConfig, *, trainable_only: bool) -> torch.optim.Optimizer:
    """AdamW over all params (backbone) or only requires_grad params (LoRA)."""
    if trainable_only:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = list(model.parameters())
    return torch.optim.AdamW(
        params,
        lr=float(cfg.train.optimizer.lr),
        weight_decay=float(cfg.train.optimizer.weight_decay),
    )


def load_initial_weights(model: PDexFlowModel, cfg: DictConfig, device: str) -> PDexFlowModel:
    """Load a pre-trained backbone into the model via the LoRA-aware loader.

    Mirrors `train_lora_real_d.py`'s original behavior. Returns the (possibly
    rewrapped) model; `load_backbone_into_lora` may return a new object so we
    propagate its return value.
    """
    ckpt_path = cfg.get("sample", {}).get("ckpt_path", None)
    if not ckpt_path:
        print("WARNING: No checkpoint path provided. Training from scratch (random initialization)!")
        return model
    print(f"Loading pre-trained backbone from {ckpt_path}...")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    return load_backbone_into_lora(model, ckpt['model'])


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available() and mps.is_built())


def resolve_inference_device(cfg_device: str) -> str:
    """Pick the best available backend for *inference / eval only*.

    Unlike training (where MPS is deliberately out of scope — see the Mac
    discussion), inference is a good fit for MPS: the sampler is pure
    forward, no GradScaler, no scipy-backed OT, and 20-step ODE sampling
    is the thing that actually benefits from a GPU on Apple Silicon.

    Rules:
      - `cpu`           → cpu (explicit override, respected)
      - `mps`           → mps if available, else cpu
      - `auto`          → cuda → mps → cpu
      - `cuda`/`cuda:N` → cuda if available; otherwise auto-fall-back to
                          mps (if available), then cpu. This lets the same
                          yaml configs (`device: cuda`) work on a Mac.

    Returns the resolved device string. Caller is expected to log both the
    requested and resolved value so silent fallbacks are visible.
    """
    req = str(cfg_device).lower().strip()
    cuda_ok = torch.cuda.is_available()
    mps_ok = _mps_available()
    if req == "cpu":
        return "cpu"
    if req == "mps":
        return "mps" if mps_ok else "cpu"
    if req == "auto":
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"
    if req.startswith("cuda"):
        if cuda_ok:
            return req
        if mps_ok:
            return "mps"
        return "cpu"
    return "cpu"
