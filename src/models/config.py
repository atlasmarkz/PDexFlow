"""Plain configuration dataclasses of PDexFlowModel."""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


@dataclass
class FlowConfig:
    # sigma_trans=0.1 matches the scale of data-side trans_1 after coord_scale=0.1
    # (typical CA coordinates within a peptide are ~tens of Å → single-digit units
    # after the scale). A unit-std prior swamps the signal at small t and pushes
    # OT cost to pick up a spurious batch translation. PFF/dflow use the same 0.1
    # scale after their coord normalization.
    sigma_trans: float = 0.1
    cfm_weight: float = 1.0
    sym_weight: float = 1.0
    seq_weight: float = 1.0
    use_ot: bool = True
    # 'huber' (default, robust) or 'l2' (squared norm, paper Eq. 4/5) for L_flow and L_sym
    loss_type: str = "huber"
    # per-component weights for velocity flow-matching (trans, rot, ang)
    cfm_w_trans: float = 1.0
    cfm_w_rot: float = 1.0
    cfm_w_ang: float = 1.0


@dataclass
class ChiLoRAConfig:
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    # "attn": q/k/v/out projections of every attention (paper Sec. IV-B); "attn_out": output projection only;
    # "ffn": both feed-forward projections.
    targets: Tuple[str, ...] = ("attn", "ffn")
    freeze_backbone: bool = False

    @classmethod
    def from_cfg(cls, cfg: Optional[Dict]) -> "ChiLoRAConfig":
        if cfg is None:
            return cls()
        if isinstance(cfg, ChiLoRAConfig):
            return cfg

        # Explicit handling: accept dict, Mapping, or OmegaConf DictConfig. Earlier
        # versions swallowed any TypeError/AttributeError and silently returned
        # disabled defaults — so a malformed config would look like "LoRA off"
        # instead of raising, which is exactly the failure mode we want to kill.
        if isinstance(cfg, Mapping):
            data = dict(cfg)
        else:
            try:
                from omegaconf import DictConfig, OmegaConf  # type: ignore
            except ImportError:
                DictConfig = None  # type: ignore
                OmegaConf = None  # type: ignore
            if DictConfig is not None and isinstance(cfg, DictConfig):
                data = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
                if not isinstance(data, dict):
                    raise TypeError(
                        f"ChiLoRAConfig.from_cfg: DictConfig resolved to {type(data)}, expected dict"
                    )
            else:
                raise TypeError(
                    f"ChiLoRAConfig.from_cfg: unsupported cfg type {type(cfg)}. "
                    f"Expected dict, Mapping, DictConfig, or ChiLoRAConfig."
                )

        targets = data.get("targets", cls.targets)
        if isinstance(targets, str):
            targets = (targets,)
        elif isinstance(targets, Iterable):
            targets = tuple(str(t) for t in targets)
        else:
            raise TypeError(
                f"ChiLoRAConfig.targets must be a string or iterable of strings, got {type(targets)}"
            )

        return cls(
            enabled=bool(data.get("enable", data.get("enabled", False))),
            rank=int(data.get("rank", cls.rank)),
            alpha=float(data.get("alpha", cls.alpha)),
            dropout=float(data.get("dropout", cls.dropout)),
            targets=targets,
            freeze_backbone=bool(data.get("freeze_backbone", cls.freeze_backbone)),
        )
