"""One-iteration train step for UniChiFlow backbone training.

Step 3 of the Trainer refactor roadmap: the inner-loop orchestration
(autocast → backward → grad_accum gate → NaN guards → opt step) is moved
out of `train_unichiflow.py` main() into a pure function so both trainers
and future subclasses can share it.

Bit-level-equivalent to the prior inline code: operation order is preserved,
skip conditions fire on the same branches, optimizer / scaler calls happen
in the same sequence. See REFACTOR_PLAN.md §四.5 for the regression contract.

Not (yet) used by LoRA trainer: it has a simpler loop (no AMP, no grad_accum,
no LR decay). Step-3 scope is backbone-only; LoRA alignment is a follow-up.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import torch
from torch import amp, nn
from torch.nn.utils import clip_grad_norm_


StepKind = Literal[
    "ok",                      # opt step applied this iteration
    "skip_loss_nonfinite",     # forward produced NaN/Inf; backward was NOT run
    "skip_grad_nonfinite",     # backward ran but grads are NaN/Inf; opt step skipped
    "accum",                   # backward accumulated; waiting for grad_accum boundary
]


@dataclass(frozen=True)
class StepConfig:
    """Immutable per-run knobs for one iteration of the training loop."""
    use_amp: bool
    amp_device: str  # "cuda" | "cpu" — matches actual training device
    grad_accum_steps: int
    max_grad_norm: float
    max_grad_value: float = 0.0  # 0 disables per-element value clipping


@dataclass
class StepOutcome:
    """What happened in this iteration. The outer loop owns bad_step counting,
    LR decay policy, smoothed loss, global_step, and logging.
    """
    kind: StepKind
    loss: float  # raw (pre-accum-divide) loss; 0.0 on skip_loss_nonfinite
    components: Dict[str, float] = field(default_factory=dict)
    grad_norm: Optional[float] = None  # populated only on kind == "ok"


def _all_grads_finite(parameters) -> bool:
    for p in parameters:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def _clip_grad_values(parameters, max_grad_value: float) -> None:
    if max_grad_value <= 0:
        return
    for p in parameters:
        if p.grad is not None:
            p.grad.clamp_(min=-max_grad_value, max=max_grad_value)


def _scalar_components(loss_dict: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in loss_dict.items():
        if torch.is_tensor(v) and v.numel() == 1:
            out[k] = v.item()
    return out


def do_train_step(
    *,
    batch: Dict[str, Any],
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    cfg: StepConfig,
    step: int,
    sym_weight_override: float,
) -> StepOutcome:
    """Forward + backward + (optionally) opt step for one iteration.

    Contract:
      - Caller zero_grad'd before the loop started; we re-zero after every
        successful opt step AND after every skip. Mid-accum steps leave grads
        in place so the next iteration keeps accumulating.
      - We NEVER touch `bad_step` or LR. Caller reads `outcome.kind` and
        decides when to increment/decay.
      - Autocast context, scaler.unscale_, clip_grad_norm_, and scaler.step
        are called in the same order as the pre-refactor inline loop.
    """
    # ---- forward (under autocast) ----
    with amp.autocast(device_type=cfg.amp_device, enabled=cfg.use_amp):
        loss_dict = model(batch, sym_weight_override=sym_weight_override)
        loss = loss_dict['loss'] / float(cfg.grad_accum_steps)

    components = _scalar_components(loss_dict)

    # ---- non-finite loss: caller must treat this like the pre-refactor
    # "non-finite loss_dict, skipping step" branch. No backward, drop grads.
    all_finite = all(
        torch.isfinite(v) for v in loss_dict.values() if torch.is_tensor(v)
    ) and torch.isfinite(loss)
    if not all_finite:
        optim.zero_grad(set_to_none=True)
        return StepOutcome(kind="skip_loss_nonfinite", loss=0.0, components=components)

    # ---- backward (always runs so grad_accum can accumulate) ----
    scaler.scale(loss).backward()

    # ---- mid-accum: don't step yet ----
    if step % cfg.grad_accum_steps != 0:
        return StepOutcome(
            kind="accum",
            loss=loss_dict['loss'].item(),
            components=components,
        )

    # ---- accum boundary: opt step with NaN guards ----
    if cfg.use_amp:
        scaler.unscale_(optim)

    # Pre-clip check: any grad non-finite → skip this accumulation entirely.
    params = list(model.parameters())
    if not _all_grads_finite(params):
        optim.zero_grad(set_to_none=True)
        if cfg.use_amp:
            scaler.update()
        return StepOutcome(
            kind="skip_grad_nonfinite",
            loss=loss_dict['loss'].item(),
            components=components,
        )
    _clip_grad_values(params, cfg.max_grad_value)

    grad = clip_grad_norm_(params, cfg.max_grad_norm)
    if not torch.isfinite(grad):
        optim.zero_grad(set_to_none=True)
        if cfg.use_amp:
            scaler.update()
        return StepOutcome(
            kind="skip_grad_nonfinite",
            loss=loss_dict['loss'].item(),
            components=components,
        )

    if cfg.use_amp:
        scaler.step(optim)
        scaler.update()
    else:
        optim.step()
    optim.zero_grad(set_to_none=True)

    return StepOutcome(
        kind="ok",
        loss=loss_dict['loss'].item(),
        components=components,
        grad_norm=float(grad),
    )
