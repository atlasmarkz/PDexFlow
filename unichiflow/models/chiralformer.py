import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from unichiflow import flow as ph_flow
from unichiflow.data import torus
from unichiflow.data import so3
from unichiflow.data import minibatch_ot
from unichiflow.data import utils as du
from unichiflow.data.pep_constants import BBHeavyAtom
from unichiflow.data.torsion import full_atom_reconstruction, get_heavyatom_mask
from unichiflow.models.chiral_gated_lora import ChiralBridgeLoRA
from unichiflow.models.geometry import construct_3d_basis
from unichiflow.models.semantic_conditioner import ESM2Conditioner, ESM2Config


def rotmat_to_6d(R: torch.Tensor) -> torch.Tensor:
    # (...,3,3) -> (...,6) : first two columns
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    # (...,6) -> (...,3,3) via Gram-Schmidt
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


@dataclass
class FlowConfig:
    sigma_trans: float = 1.0
    cfm_weight: float = 1.0
    sym_weight: float = 1.0
    seq_weight: float = 1.0
    use_ot: bool = True


@dataclass
class ChiLoRAConfig:
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    targets: Tuple[str, ...] = ("attn_out", "ffn")
    freeze_backbone: bool = False

    @classmethod
    def from_cfg(cls, cfg: Optional[Dict]) -> "ChiLoRAConfig":
        if cfg is None:
            return cls()
        if isinstance(cfg, ChiLoRAConfig):
            return cfg
        try:
            data = {k: v for k, v in cfg.items()}
        except Exception:
            try:
                data = dict(cfg)
            except Exception:
                data = {}

        targets = data.get("targets", cls.targets)
        if isinstance(targets, str):
            targets = (targets,)
        elif isinstance(targets, Iterable):
            targets = tuple(str(t) for t in targets)
        else:
            targets = cls.targets

        return cls(
            enabled=bool(data.get("enable", data.get("enabled", False))),
            rank=int(data.get("rank", cls.rank)),
            alpha=float(data.get("alpha", cls.alpha)),
            dropout=float(data.get("dropout", cls.dropout)),
            targets=targets,
            freeze_backbone=bool(data.get("freeze_backbone", cls.freeze_backbone)),
        )


class ChiLoRALinear(nn.Module):
    """LoRA wrapper for Linear with optional backbone freezing."""

    def __init__(self, base: nn.Linear, cfg: ChiLoRAConfig):
        super().__init__()
        if cfg.rank <= 0:
            raise ValueError("ChiLoRA rank must be positive")

        self.base = base
        self.rank = int(cfg.rank)
        self.scaling = float(cfg.alpha) / float(self.rank)
        self.dropout = nn.Dropout(float(cfg.dropout)) if cfg.dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.zeros((self.rank, base.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, self.rank)))
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

        self._gate: Optional[torch.Tensor] = None

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        if cfg.freeze_backbone:
            for p in self.base.parameters():
                p.requires_grad = False

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def set_gate(self, gate: Optional[torch.Tensor]):
        # gate is broadcastable to (B,L,1) or (B,1,1); None disables gating
        self._gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        gate_scale = torch.sigmoid(self.gate_logit)
        if self._gate is not None:
            lora_out = lora_out * self._gate
        else:
            lora_out = lora_out * 0.0
        return base_out + self.scaling * lora_out * gate_scale


class UniChiFlowModel(nn.Module):
    """Core UniChiFlow with optional ChiLoRA adapters.

    Learns to predict x1 from (xt, t, chirality_token, pocket context) using a transformer.
    State = (seq, CA-translation, backbone-rotation, torsion(psi+chi)).
    """

    def __init__(self, d_model: int = 256, n_layers: int = 6, n_heads: int = 8, dropout: float = 0.1, max_len: int = 512, aatype_vocab: int = 21,
                 sigma_trans: float = 1.0, chi_lora: Optional[Dict] = None, esm: Optional[Dict] = None, cfm_weight: float = 1.0, sym_weight: float = 1.0,
                 seq_weight: float = 1.0, angle_dim: int = 5, use_ot: bool = True):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.aatype_vocab = int(aatype_vocab)
        self.flow_cfg = FlowConfig(
            sigma_trans=float(sigma_trans),
            cfm_weight=float(cfm_weight),
            sym_weight=float(sym_weight),
            seq_weight=float(seq_weight),
            use_ot=bool(use_ot),
        )
        self.chi_lora_cfg = ChiLoRAConfig.from_cfg(chi_lora)

        self.seq_emb = nn.Embedding(self.aatype_vocab + 2, self.d_model)  # +pad
        self.pos_emb = nn.Embedding(self.max_len + 8, self.d_model)
        self.gen_mask_emb = nn.Embedding(2, self.d_model)
        self.chirality_emb = nn.Embedding(2, self.d_model)

        self.angle_dim = int(angle_dim)

        self.trans_in = nn.Linear(3, self.d_model, bias=False)
        self.rot_in = nn.Linear(6, self.d_model, bias=False)
        self.ang_in = nn.Linear(self.angle_dim * 2, self.d_model)
        self.t_in = nn.Linear(self.d_model, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=int(n_heads), dim_feedforward=self.d_model * 4, dropout=float(dropout),
                                                   batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers), enable_nested_tensor=False)
        self._chi_lora_modules: Tuple[ChiLoRALinear, ...] = tuple()

        self.esm_conditioner: Optional[ESM2Conditioner] = None
        self.esm_proj: Optional[nn.Module] = None
        self.esm_adapter: Optional[ChiralBridgeLoRA] = None

        if self.chi_lora_cfg.enabled:
            self._apply_chi_lora()

        # === ESM/Adapter维度自动推断修正 ===
        if esm and bool(esm.get("enable", False)):
            esm_cfg = ESM2Config(
                model=str(esm.get("model", ESM2Config.model)),
                repr_layer=int(esm.get("repr_layer", -1)),
                requires_grad=bool(esm.get("requires_grad", False)),
            )
            self.esm_conditioner = ESM2Conditioner(esm_cfg)
            esm_hidden_size = self.esm_conditioner.hidden_size
            self.esm_proj = nn.Linear(esm_hidden_size, self.d_model)
            use_adapter = bool(esm.get("use_adapter", True))
            if use_adapter:
                # Adapter应为nn.Linear(d_model, d_model)，保证输入输出都是d_model
                self.esm_adapter = ChiralBridgeLoRA(
                    nn.Linear(self.d_model, self.d_model),
                    rank=int(esm.get("rank", 8)),
                    alpha=float(esm.get("alpha", 8.0)),
                    dropout=float(esm.get("dropout", 0.0)),
                )

        self.out_trans = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 3, bias=False))
        self.out_rot6d = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 6, bias=False))
        self.out_angles = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.angle_dim))
        self.out_seq = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.aatype_vocab))

    def _apply_chi_lora(self):
        targets: Set[str] = set(self.chi_lora_cfg.targets)
        modules = []
        for layer in self.trunk.layers:
            if "ffn" in targets:
                layer.linear1 = ChiralBridgeLoRA(layer.linear1, rank=self.chi_lora_cfg.rank, alpha=self.chi_lora_cfg.alpha, dropout=self.chi_lora_cfg.dropout)
                layer.linear2 = ChiralBridgeLoRA(layer.linear2, rank=self.chi_lora_cfg.rank, alpha=self.chi_lora_cfg.alpha, dropout=self.chi_lora_cfg.dropout)
                modules.extend([layer.linear1, layer.linear2])
            if "attn_out" in targets:
                layer.self_attn.out_proj = ChiralBridgeLoRA(layer.self_attn.out_proj, rank=self.chi_lora_cfg.rank, alpha=self.chi_lora_cfg.alpha, dropout=self.chi_lora_cfg.dropout)
                modules.append(layer.self_attn.out_proj)

        self._chi_lora_modules = tuple(modules)

        if self.chi_lora_cfg.freeze_backbone:
            for name, param in self.named_parameters():
                if "lora_" not in name:
                    param.requires_grad = False

    def _set_chi_lora_gate(self, gate: Optional[torch.Tensor]):
        for m in self._chi_lora_modules:
            m.set_gate(gate)

    def _clear_chi_lora_gate(self):
        for m in self._chi_lora_modules:
            m.set_gate(None)

    def _embed_t(self, t: torch.Tensor, L: int) -> torch.Tensor:
        # t: (B,1) -> (B,L,d)
        te = du.get_time_embedding(t[:, 0], self.d_model, max_positions=2048)
        return te[:, None, :].repeat(1, L, 1)

    @staticmethod
    def _angles_to_sincos(angles: torch.Tensor) -> torch.Tensor:
        # (B,L,5) -> (B,L,10)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    @staticmethod
    def _random_quat(shape, device, dtype):
        q = torch.randn(*shape, 4, device=device, dtype=dtype)
        return F.normalize(q, dim=-1)

    def corrupt_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create interpolants and endpoints for PH-Flow with optional minibatch OT alignment."""
        aa_1 = batch['aa']
        pos = batch['pos_heavyatom']
        gen_mask = batch['generate_mask'].bool()
        res_mask = batch['res_mask'].bool()

        rot_1 = construct_3d_basis(pos[:, :, BBHeavyAtom.CA], pos[:, :, BBHeavyAtom.C], pos[:, :, BBHeavyAtom.N])
        trans_1 = pos[:, :, BBHeavyAtom.CA]
        ang_1 = batch['torsion_angle']

        B, L = aa_1.shape
        device = aa_1.device

        t = torch.rand((B, 1), device=device)
        t = t * 0.99 + 0.005

        # noise
        trans_0 = torch.randn((B, L, 3), device=device) * self.flow_cfg.sigma_trans
        rot_0 = so3.quat_to_rotmat(self._random_quat((B, L), device=device, dtype=trans_1.dtype))
        ang_0 = torus.tor_random_uniform(B, L, ang_1.shape[-1], device=device, dtype=ang_1.dtype)

        pep_mask = gen_mask & res_mask
        if self.flow_cfg.use_ot and pep_mask.any():
            # OT aligns noisy samples to data within peptide residues, improving straightness of flow lines.
            for b in range(B):
                idx = torch.nonzero(pep_mask[b], as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                z0 = (trans_0[b:b + 1, idx], rot_0[b:b + 1, idx], ang_0[b:b + 1, idx])
                z1 = (trans_1[b:b + 1, idx], rot_1[b:b + 1, idx], ang_1[b:b + 1, idx])
                cost = minibatch_ot.pairwise_cost(z0, z1)
                perm = minibatch_ot.solve_ot(cost)[0]
                trans_0[b, idx] = z0[0][0, perm]
                rot_0[b, idx] = z0[1][0, perm]
                ang_0[b, idx] = z0[2][0, perm]

        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        # t should be broadcastable to (B,L,1) inside slerp; using two new axes would
        # incorrectly broadcast (B,L,4) -> (1,B,L,4) and create an extra B dimension.
        rot_t = so3.geodesic_t(t[..., None], rot_1, rot_0)
        ang_t = torus.tor_geodesic_t(t[..., None], ang_1, ang_0)

        # keep receptor fixed (non-generated residues)
        trans_t = torch.where(gen_mask[..., None], trans_t, trans_1)
        rot_t = torch.where(gen_mask[..., None, None], rot_t, rot_1)
        ang_t = torch.where(gen_mask[..., None], ang_t, ang_1)

        # sequence corruption on peptide only
        seq_t = aa_1.clone()
        corruption = (torch.rand((B, L), device=device) < (1 - t)) & gen_mask & res_mask
        seq_t[corruption] = du.MASK_TOKEN_INDEX
        # set padding positions to PAD
        seq_t = torch.where(res_mask, seq_t, torch.full_like(seq_t, du.MASK_TOKEN_INDEX))

        return t, seq_t, trans_t, rot_t, ang_t, aa_1, trans_1, rot_1, ang_1, trans_0, rot_0, ang_0

    def predict_x1(self, batch: Dict[str, torch.Tensor], t: torch.Tensor, seq_t: torch.Tensor, trans_t: torch.Tensor, rot_t: torch.Tensor, ang_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        def _squeeze_last_if_one(x: torch.Tensor, dims, name: str) -> torch.Tensor:
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
            for d in dims:
                if x.ndim > 0 and x.shape[d] == 1:
                    x = x.squeeze(d)
            return x

        # Normalize common singleton extra dims from cached/legacy datasets.
        seq_t = _squeeze_last_if_one(seq_t, dims=[-1], name='seq_t')
        trans_t = _squeeze_last_if_one(trans_t, dims=[-2], name='trans_t')
        ang_t = _squeeze_last_if_one(ang_t, dims=[-2], name='ang_t')
        res_mask = _squeeze_last_if_one(batch['res_mask'], dims=[-1], name='res_mask').bool()
        gen_mask = _squeeze_last_if_one(batch['generate_mask'], dims=[-1], name='generate_mask').long()

        if seq_t.ndim != 2:
            raise ValueError(f"Expected seq_t to have shape (B,L), got {tuple(seq_t.shape)}")
        if res_mask.ndim != 2:
            raise ValueError(f"Expected res_mask to have shape (B,L), got {tuple(res_mask.shape)}")
        if gen_mask.ndim != 2:
            raise ValueError(f"Expected generate_mask to have shape (B,L), got {tuple(gen_mask.shape)}")

        if trans_t.ndim != 3 or trans_t.shape[-1] != 3:
            raise ValueError(f"Expected trans_t to have shape (B,L,3), got {tuple(trans_t.shape)}")
        if ang_t.ndim != 3:
            raise ValueError(f"Expected ang_t to have shape (B,L,A), got {tuple(ang_t.shape)}")
        if rot_t.ndim != 4 or rot_t.shape[-2:] != (3, 3):
            raise ValueError(f"Expected rot_t to have shape (B,L,3,3), got {tuple(rot_t.shape)}")

        B, L = seq_t.shape

        # positional ids: clamp into max_len
        pos_ids = batch.get('res_nb', torch.arange(L, device=seq_t.device)[None].repeat(B, 1))
        if isinstance(pos_ids, torch.Tensor):
            pos_ids = _squeeze_last_if_one(pos_ids, dims=[-1], name='res_nb')
        pos_ids = pos_ids.clamp(min=0, max=self.max_len + 7)
        if not isinstance(pos_ids, torch.Tensor) or pos_ids.ndim != 2:
            raise ValueError(f"Expected res_nb/pos_ids to have shape (B,L), got {type(pos_ids)} with shape {getattr(pos_ids, 'shape', None)}")

        x = (
            self.seq_emb(seq_t.clamp(min=0, max=self.aatype_vocab + 1))
            + self.pos_emb(pos_ids)
            + self.gen_mask_emb(gen_mask)
        )

        chir = batch.get('chirality', torch.zeros((B,), device=seq_t.device, dtype=torch.long))
        if isinstance(chir, torch.Tensor) and chir.ndim == 2 and chir.shape[-1] == 1:
            chir = chir.squeeze(-1)
        if not isinstance(chir, torch.Tensor) or chir.ndim != 1:
            raise ValueError(f"Expected chirality to have shape (B,), got {type(chir)} with shape {getattr(chir, 'shape', None)}")
        x = x + self.chirality_emb(chir)[:, None, :]

        esm_cond = batch.get('esm_cond', None)
        if esm_cond is None and self.esm_conditioner is not None:
            with torch.no_grad():
                esm_cond, _ = self.esm_conditioner(seq_t, res_mask)
        if esm_cond is not None:
            if not isinstance(esm_cond, torch.Tensor):
                raise TypeError(f"esm_cond must be a torch.Tensor, got {type(esm_cond)}")
            if esm_cond.ndim == 4 and esm_cond.shape[-2] == 1:
                esm_cond = esm_cond.squeeze(-2)
            if esm_cond.shape[:2] != (B, L):
                raise ValueError(f"esm_cond must have shape (B,L,*) matching the sequence length, got {tuple(esm_cond.shape)} vs (B={B},L={L})")
            # === 先用esm_proj投影到d_model，再送入esm_adapter（LoRA） ===
            if self.esm_proj is not None:
                esm_cond = self.esm_proj(esm_cond)  # [B, L, hidden_size] -> [B, L, d_model]
            if self.esm_adapter is not None:
                esm_cond = self.esm_adapter(esm_cond, chirality_token=chir)  # [B, L, d_model] -> [B, L, d_model]
            if esm_cond.shape[-1] != self.d_model:
                raise ValueError(f"esm_cond last dim must match d_model={self.d_model}, got {esm_cond.shape[-1]}")
            x = x + esm_cond

        if ang_t.shape[-1] < self.angle_dim:
            pad = torch.zeros((*ang_t.shape[:-1], self.angle_dim - ang_t.shape[-1]), device=ang_t.device, dtype=ang_t.dtype)
            ang_t = torch.cat([ang_t, pad], dim=-1)
        elif ang_t.shape[-1] > self.angle_dim:
            ang_t = ang_t[..., : self.angle_dim]

        x = x + self.trans_in(trans_t)
        x = x + self.rot_in(rotmat_to_6d(rot_t))
        x = x + self.ang_in(self._angles_to_sincos(ang_t))
        x = x + self.t_in(self._embed_t(t, L))

        # Defensive: Transformer expects (B,L,E) with batch_first=True
        if x.ndim == 4 and x.shape[-2] == 1:
            x = x.squeeze(-2)
        if x.ndim != 3:
            raise ValueError(
                "Transformer input x must be 3D (B,L,E). "
                f"Got x={tuple(x.shape)}. "
                f"seq_t={tuple(seq_t.shape)}, pos_ids={tuple(pos_ids.shape)}, gen_mask={tuple(gen_mask.shape)}, "
                f"trans_t={tuple(trans_t.shape)}, rot_t={tuple(rot_t.shape)}, ang_t={tuple(ang_t.shape)}, res_mask={tuple(res_mask.shape)}"
            )

        # transformer
        if self.chi_lora_cfg.enabled:
            pep_mask = (gen_mask.bool() & res_mask.bool()).unsqueeze(-1)  # (B,L,1)
            gate = batch.get('chirality', torch.zeros((B,), device=seq_t.device, dtype=torch.long))
            if isinstance(gate, torch.Tensor) and gate.ndim == 2 and gate.shape[-1] == 1:
                gate = gate.squeeze(-1)
            gate = gate.view(B, 1, 1).to(seq_t.device, dtype=x.dtype) * pep_mask.to(x.dtype)
            self._set_chi_lora_gate(gate)
        else:
            gate = None

        h = self.trunk(x, src_key_padding_mask=(~res_mask))

        if gate is not None:
            self._clear_chi_lora_gate()

        pred_trans_1 = self.out_trans(h)
        pred_rot_1 = rot6d_to_rotmat(self.out_rot6d(h))
        pred_ang_1 = self.out_angles(h) % (2 * math.pi)
        pred_seq_logits = self.out_seq(h)

        return {
            'pred_trans_1': pred_trans_1,
            'pred_rot_1': pred_rot_1,
            'pred_ang_1': pred_ang_1,
            'pred_seq_logits': pred_seq_logits,
        }

    def forward(self, batch: Dict[str, torch.Tensor], sym_weight_override: Optional[float] = None) -> Dict[str, torch.Tensor]:
        t, seq_t, trans_t, rot_t, ang_t, aa_1, trans_1, rot_1, ang_1, trans_0, rot_0, ang_0 = self.corrupt_batch(batch)
        out = self.predict_x1(batch, t, seq_t, trans_t, rot_t, ang_t)

        res_mask = batch['res_mask'].bool()
        gen_mask = batch['generate_mask'].bool()
        mask = res_mask & gen_mask

        # Changed to State-Matching Loss (x1-prediction) to avoid 1/(1-t) singularity.
        # v_true -> target states (trans_1, rot_1, ang_1)
        # v_pred -> predicted states (from model output)
        pred_states = (out['pred_trans_1'], out['pred_rot_1'], out['pred_ang_1'])
        target_states = (trans_1, rot_1, ang_1)
        
        cfm_loss = ph_flow.flow_matching_loss(pred_states, target_states, mask)

        sym_loss = ph_flow.compute_symmetry_loss(
            predict_fn=lambda b, t_in, seq_in, trans_in, rot_in, ang_in: self.predict_x1(b, t_in, seq_in, trans_in, rot_in, ang_in),
            batch=batch,
            t=t,
            seq_t=seq_t,
            trans_t=trans_t,
            rot_t=rot_t,
            ang_t=ang_t,
            mask=mask,
            pred_out=out,
            v_pred=None,
        )

        # CE only on peptide residues
        if mask.any():
            seq_loss = F.cross_entropy(out['pred_seq_logits'][mask], aa_1[mask].clamp(max=self.aatype_vocab - 1))
        else:
            seq_loss = torch.tensor(0.0, device=aa_1.device)
        
        # Apply weights with optional override for symmetry
        w_sym = sym_weight_override if sym_weight_override is not None else self.flow_cfg.sym_weight
        total = self.flow_cfg.cfm_weight * cfm_loss + w_sym * sym_loss + self.flow_cfg.seq_weight * seq_loss
        return {
            'loss': total,
            'cfm_loss': cfm_loss,
            'sym_loss': sym_loss,
            'seq_loss': seq_loss,
        }

    @torch.no_grad()
    def decode_to_heavyatom15(self, aa: torch.Tensor, trans: torch.Tensor, rotmats: torch.Tensor, angles: torch.Tensor, chirality: torch.Tensor, peptide_mask: torch.Tensor, hard_constraints: bool = True):
        """Geometric decoder with optional hard chirality enforcement.

        When hard_constraints is True (default), uses torsion-based reconstruction and applies
        mirrored local frames for D-mode residues to guarantee chirality correctness. When False,
        it skips the reflection, serving as a soft/ablated decoder for chiral error analysis.
        """
        B, L = aa.shape
        device = aa.device
        # reflection on local z-axis (M_z)
        M = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=device, dtype=rotmats.dtype))

        rotmats_dec = rotmats.clone()
        chir = chirality.view(B, 1, 1, 1).to(rotmats.dtype)
        pep = peptide_mask.view(B, L, 1, 1).to(rotmats.dtype)
        base = torch.eye(3, device=device, dtype=rotmats.dtype)
        if hard_constraints:
            # Enforce chirality by reflecting D residues in local frame
            rotmats_dec = rotmats_dec @ (base + chir * pep * (M - base))

        pos14, _, _ = full_atom_reconstruction(R_bb=rotmats_dec, t_bb=trans, angles=angles, aa=aa)
        pos15 = F.pad(pos14, pad=(0, 0, 0, 15 - 14), value=0.0)
        mask15 = get_heavyatom_mask(aa)
        return pos15, mask15

    @torch.no_grad()
    def sample(self, batch: Dict[str, torch.Tensor], num_steps: int = 30, angle_purify: bool = True):
        device = batch['aa'].device
        aa_gt = batch['aa']
        pos = batch['pos_heavyatom']
        gen_mask = batch['generate_mask'].bool()
        res_mask = batch['res_mask'].bool()
        B, L = aa_gt.shape

        # fixed context (receptor)
        rot_1 = construct_3d_basis(pos[:, :, BBHeavyAtom.CA], pos[:, :, BBHeavyAtom.C], pos[:, :, BBHeavyAtom.N])
        trans_1 = pos[:, :, BBHeavyAtom.CA]
        ang_1 = batch['torsion_angle']

        # init state
        t0 = torch.zeros((B, 1), device=device)
        trans = torch.where(gen_mask[..., None], torch.randn((B, L, 3), device=device) * self.flow_cfg.sigma_trans, trans_1)
        rot = torch.where(gen_mask[..., None, None], so3.quat_to_rotmat(self._random_quat((B, L), device=device, dtype=trans.dtype)), rot_1)
        ang = torch.where(gen_mask[..., None], torus.tor_random_uniform(B, L, ang_1.shape[-1], device=device, dtype=ang_1.dtype), ang_1)
        seq = torch.where(gen_mask, torch.full_like(aa_gt, du.MASK_TOKEN_INDEX), aa_gt)
        seq = torch.where(res_mask, seq, torch.full_like(seq, du.MASK_TOKEN_INDEX))

        dt = 1.0 / float(num_steps)
        for k in range(num_steps):
            t = torch.full((B, 1), (k + 1) / float(num_steps), device=device)
            pred = self.predict_x1(batch, t, seq, trans, rot, ang)

            # update peptide only
            pep = gen_mask & res_mask
            if pep.any():
                trans_new = (1 - dt) * trans + dt * pred['pred_trans_1']
                rot_new = so3.geodesic_t(torch.full((B, 1, 1), dt, device=device), pred['pred_rot_1'], rot)
                ang_new = torus.tor_geodesic_t(torch.full((B, 1, 1), dt, device=device), pred['pred_ang_1'], ang)

                trans = torch.where(pep[..., None], trans_new, trans)
                rot = torch.where(pep[..., None, None], rot_new, rot)
                ang = torch.where(pep[..., None], ang_new, ang)

                seq_pred = torch.argmax(pred['pred_seq_logits'], dim=-1)
                seq = torch.where(pep, seq_pred, seq)

            if angle_purify:
                ang = ang % (2 * math.pi)

        return {
            'seq': seq,
            'trans': trans,
            'rotmats': rot,
            'angles': ang,
        }
