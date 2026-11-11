import contextlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import flow as ph_flow
from src.data import torus
from src.data import so3
from src.data import minibatch_ot
from src.data import utils as du
from src.data.pep_constants import BBHeavyAtom
from src.data.torsion import full_atom_reconstruction, get_heavyatom_mask
from src.models.chiral_gated_lora import ChiralBridgeLoRA
from src.models.geometric_transformer import GeometricTransformer
from src.models.geometry import construct_3d_basis
from src.models.semantic_conditioner import ESM2Conditioner, ESM2Config


def rotmat_to_6d(R: torch.Tensor) -> torch.Tensor:
    # (...,3,3) -> (...,6) : first two columns
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    # (...,6) -> (...,3,3) via Gram-Schmidt
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    # eps=1e-6 (not the default 1e-12): under fp16/AMP the head can emit near-zero
    # 6D vectors early in training, which F.normalize then turns into NaN when the
    # default eps underflows. Matches the Gram–Schmidt guard used in PFF.
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def compute_axial_features(trans: torch.Tensor, rot6d: torch.Tensor) -> torch.Tensor:
    """Compute axial (pseudo)vector features from polar vectors.

    Following PepMirror (Yang et al., 2026), we construct three types of axial
    features from the polar vector inputs (translations and rotation columns):

    1. Cross product: u x v — invariant under parity (P(u)xP(v) = (-u)x(-v) = uxv)
    2. Commutator: (u.v)(u x v) — captures higher-frequency chirality information
    3. Triple product projection: ((w.(u x v)) / ||u x v||) * (u x v) — scalar
       triple product modulated axial direction

    These axial features break the E(3) equivariance that causes chirality blindness
    in standard scalarization-based models, making the network SE(3)-sensitive
    (able to distinguish L from D enantiomers).

    Args:
        trans: (B, L, 3) translation vectors (CA positions).
        rot6d: (B, L, 6) 6D rotation representation (first two columns of R).

    Returns:
        (B, L, 9) concatenated axial features [cross, commutator, triple_proj].
    """
    u = trans  # (B, L, 3)
    v = rot6d[..., 0:3]  # first column of rotation matrix
    w = rot6d[..., 3:6]  # second column of rotation matrix

    # Axial feature 1: cross product (parity-invariant)
    cross = torch.cross(u, v, dim=-1)  # (B, L, 3)

    # Axial feature 2: commutator (u.v)(u x v)
    dot_uv = (u * v).sum(dim=-1, keepdim=True)  # (B, L, 1)
    commutator = dot_uv * cross  # (B, L, 3)

    # Axial feature 3: triple product projection
    triple_scalar = (w * cross).sum(dim=-1, keepdim=True)  # (B, L, 1)
    cross_norm = torch.norm(cross, dim=-1, keepdim=True).clamp(min=1e-8)
    triple_proj = (triple_scalar / cross_norm) * (cross / cross_norm)  # (B, L, 3)

    return torch.cat([cross, commutator, triple_proj], dim=-1)  # (B, L, 9)


class AFIModule(nn.Module):
    """Axial Feature Injection (AFI) module.

    Injects axial vector features into the polar vector feature stream via
    channel-wise linear mixing, as proposed in PepMirror:

        v_tilde = A^T v_polar + B^T v_axial

    where A, B are learnable mixing coefficients. This makes the model
    chirality-aware (SE(3)-sensitive instead of E(3)-equivariant), enabling
    zero-shot cross-chirality generalization from L-peptide training data
    to D-peptide generation.

    The module is inserted before every FFN layer in the transformer,
    modifying the hidden state to include chirality-sensitive features.
    """

    def __init__(self, d_model: int, axial_dim: int = 9):
        super().__init__()
        self.d_model = d_model
        self.axial_dim = axial_dim
        # Learnable mixing: project axial features to d_model, then mix
        self.axial_proj = nn.Linear(axial_dim, d_model, bias=False)
        # Initialize with small weights so AFI starts near identity (polar-only)
        nn.init.zeros_(self.axial_proj.weight)
        # Gate that controls AFI strength (starts at 0, learns to open)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, h: torch.Tensor, trans: torch.Tensor, rot6d: torch.Tensor) -> torch.Tensor:
        """Inject axial features into the hidden state.

        Args:
            h: (B, L, d_model) transformer hidden state.
            trans: (B, L, 3) translation vectors.
            rot6d: (B, L, 6) 6D rotation representation.

        Returns:
            (B, L, d_model) hidden state with axial feature injection.
        """
        axial = compute_axial_features(trans, rot6d)  # (B, L, 9)
        axial_proj = self.axial_proj(axial)  # (B, L, d_model)
        gate = torch.sigmoid(self.gate_logit)
        return h + gate * axial_proj


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
    # 'minibatch': Mini-ROT of paper Eq. 13 (assignment between the B priors and B data samples of a minibatch)
    # 'residue':   legacy within-peptide residue-level assignment
    ot_mode: str = "minibatch"
    # 'huber' (default, robust) or 'l2' (squared norm, paper Eq. 4/5) for L_flow and L_sym
    loss_type: str = "huber"
    # per-component weights for velocity flow-matching (trans, rot, ang)
    cfm_w_trans: float = 1.0
    cfm_w_rot: float = 1.0
    cfm_w_ang: float = 1.0
    # Automatic loss weighting
    auto_loss_weighting: bool = False


@dataclass
class AFIConfig:
    # Axial Feature Injection is NOT part of the PDexFlow paper (borrowed from PepMirror); opt-in only.
    enabled: bool = False
    axial_dim: int = 9  # cross(3) + commutator(3) + triple_proj(3)


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


class PDexFlowModel(nn.Module):
    """Core PDexFlow with optional ChiLoRA adapters.

    Learns to predict x1 from (xt, t, chirality_token, pocket context) using a transformer.
    State = (seq, CA-translation, backbone-rotation, torsion(psi+chi)).
    """

    def __init__(self, d_model: int = 256, n_layers: int = 6, n_heads: int = 8, dropout: float = 0.1, max_len: int = 512, aatype_vocab: int = 21,
                 sigma_trans: float = 0.1, chi_lora: Optional[Dict] = None, esm: Optional[Dict] = None, cfm_weight: float = 1.0, sym_weight: float = 1.0,
                 seq_weight: float = 1.0, angle_dim: int = 5, use_ot: bool = True,
                 cfm_w_trans: float = 1.0, cfm_w_rot: float = 1.0, cfm_w_ang: float = 1.0,
                 coord_scale: float = 0.1, afi: Optional[Dict] = None,
                 trunk_type: str = "geometric", n_rbf: int = 16, ot_mode: str = "minibatch", loss_type: str = "huber"):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.aatype_vocab = int(aatype_vocab)
        # Single-source-of-truth for coordinate unit conversion: PDB angstroms -> model units.
        # Applied in corrupt_batch and sample (entry points that consume raw batch['pos_heavyatom']).
        # Training/inference scripts must NOT pre-scale batch['pos_heavyatom'] themselves.
        self.coord_scale = float(coord_scale)
        self.flow_cfg = FlowConfig(
            sigma_trans=float(sigma_trans),
            cfm_weight=float(cfm_weight),
            sym_weight=float(sym_weight),
            seq_weight=float(seq_weight),
            use_ot=bool(use_ot),
            ot_mode=str(ot_mode),
            loss_type=str(loss_type),
            cfm_w_trans=float(cfm_w_trans),
            cfm_w_rot=float(cfm_w_rot),
            cfm_w_ang=float(cfm_w_ang),
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

        if trunk_type not in ("geometric", "joint"):
            raise ValueError(f"trunk_type must be 'geometric' (paper) or 'joint' (legacy), got {trunk_type!r}")
        self.trunk_type = trunk_type
        if trunk_type == "geometric":
            # Paper Sec. IV-C: RBF-biased peptide self-attention + receptor cross-attention.
            self.trunk = GeometricTransformer(self.d_model, int(n_layers), int(n_heads), dropout=float(dropout), n_rbf=int(n_rbf))
        else:
            # Legacy trunk: plain self-attention over the concatenated receptor+peptide sequence.
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=int(n_heads), dim_feedforward=self.d_model * 4, dropout=float(dropout),
                                                       batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers), enable_nested_tensor=False)
        self._chi_lora_modules: Tuple[ChiLoRALinear, ...] = tuple()

        # === Axial Feature Injection (AFI) ===
        # AFI modules are inserted before each transformer layer to inject
        # chirality-sensitive axial features into the hidden state.
        afi_cfg = AFIConfig()
        if afi is not None and isinstance(afi, Mapping):
            afi_cfg = AFIConfig(enabled=bool(afi.get("enable", afi.get("enabled", True))))
        elif afi is not None and hasattr(afi, "get"):
            afi_cfg = AFIConfig(enabled=bool(afi.get("enable", afi.get("enabled", True))))
        self.afi_cfg = afi_cfg
        self.afi_modules: nn.ModuleList = nn.ModuleList()
        if self.afi_cfg.enabled:
            for _ in range(int(n_layers)):
                self.afi_modules.append(AFIModule(self.d_model, axial_dim=self.afi_cfg.axial_dim))

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
                self.esm_adapter = ChiralBridgeLoRA(
                    nn.Linear(self.d_model, self.d_model),
                    rank=int(esm.get("rank", 8)),
                    alpha=float(esm.get("alpha", 8.0)),
                    dropout=float(esm.get("dropout", 0.0)),
                    gate_init=0.0,
                )

        self.out_trans = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 3, bias=False))
        self.out_rot6d = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 6, bias=False))
        self.out_angles = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.angle_dim))
        self.out_seq = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.aatype_vocab))

        # Learnable loss weights (log variances) for Uncertainty Weighting
        # log_vars[0]: cfm, log_vars[1]: sym, log_vars[2]: seq
        if self.flow_cfg.auto_loss_weighting:
            self.loss_log_vars = nn.Parameter(torch.zeros(3))
        else:
            self.loss_log_vars = None

    def _apply_chi_lora(self):
        """Wrap the trunk projections with Chiral-Bridge LoRA (paper Eq. 7-8).

        targets: "attn"      -> q/k/v/out projections of every attention module (self + cross),
                 "attn_out"  -> output projection only,
                 "ffn"       -> both feed-forward projections.
        The legacy ``joint`` trunk uses nn.MultiheadAttention (packed q/k/v), where only ``out_proj`` can be wrapped.
        """
        targets: Set[str] = set(self.chi_lora_cfg.targets)
        cfg = self.chi_lora_cfg
        modules = []

        def wrap(parent, attr):
            mod = ChiralBridgeLoRA(getattr(parent, attr), rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout, gate_init=0.0)
            setattr(parent, attr, mod)
            modules.append(mod)

        for layer in self.trunk.layers:
            if "ffn" in targets:
                wrap(layer, "linear1")
                wrap(layer, "linear2")
            attn_modules = [layer.self_attn] + ([layer.cross_attn] if hasattr(layer, "cross_attn") else [])
            for att in attn_modules:
                if "attn" in targets and hasattr(att, "q_proj"):
                    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                        wrap(att, name)
                elif "attn" in targets or "attn_out" in targets:
                    wrap(att, "out_proj")

        self._chi_lora_modules = tuple(modules)

        if self.trunk_type == "joint" and hasattr(torch.backends, "mha"):
            # nn.TransformerEncoderLayer's eval/no_grad fast path reads linear1.weight etc. directly and would
            # silently bypass the LoRA wrappers at inference time.
            torch.backends.mha.set_fastpath_enabled(False)

        if self.chi_lora_cfg.freeze_backbone:
            for name, param in self.named_parameters():
                is_lora = "lora_" in name
                is_chiral_gate = name.endswith("gate_logit") and "afi_modules" not in name  # gamma in G(c)=c*sigmoid(gamma)
                if not (is_lora or is_chiral_gate):
                    param.requires_grad = False

    def _set_chi_lora_gate(self, gate: Optional[torch.Tensor]):
        # Guard against overlapping forward passes that forget to clear the previous gate.
        # The two predict_x1 calls in forward()/compute_symmetry_loss run sequentially,
        # so gate should be None whenever we set a new one; assert that invariant.
        for m in self._chi_lora_modules:
            if m._gate is not None and gate is not None:
                raise RuntimeError("ChiLoRA gate is already set; previous forward did not clear it")
            m.set_gate(gate)

    def _clear_chi_lora_gate(self):
        for m in self._chi_lora_modules:
            m.set_gate(None)

    @contextlib.contextmanager
    def _chi_lora_gate(self, gate: Optional[torch.Tensor]):
        """Scope a ChiLoRA gate to a single forward pass and always clear it on exit."""
        self._set_chi_lora_gate(gate)
        try:
            yield
        finally:
            self._clear_chi_lora_gate()

    @staticmethod
    def _safe_padding_mask(res_mask: torch.Tensor) -> torch.Tensor:
        """Build Transformer src_key_padding_mask (True = pad) from res_mask (True = valid).

        Guards against rows where every position is padded: nn.TransformerEncoder's
        softmax degenerates to 0/0 -> NaN if an entire row is masked out. When we
        detect such a row we flip position 0 to valid; the corresponding token is
        meaningless (it's a padding slot), but it keeps softmax numerically defined
        and downstream loss masks (res_mask & generate_mask) still exclude it.
        """
        key_pad = ~res_mask
        all_pad_rows = key_pad.all(dim=-1)
        if all_pad_rows.any():
            key_pad = key_pad.clone()
            key_pad[all_pad_rows, 0] = False
        return key_pad

    def _esm_tokens(self, seq_t: torch.Tensor, gen_mask: torch.Tensor) -> torch.Tensor:
        """Tokens fed to ESM-2. Paper: the frozen pLM encodes the *receptor*; hide the noisy peptide tokens."""
        if self.trunk_type != "geometric":
            return seq_t
        return torch.where(gen_mask.bool(), torch.full_like(seq_t, du.MASK_TOKEN_INDEX), seq_t)

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

    @staticmethod
    def _center_noise_on_pep(trans_noise: torch.Tensor, pep_mask: torch.Tensor) -> torch.Tensor:
        """Zero-mean translational noise over peptide residues.

        PFF / dflow both initialize the flow from a batch-centered Gaussian — otherwise
        a random noise batch has a non-zero CoM, so the OT cost picks up a global
        translation bias and flow lines are no longer straight. We apply the same
        centering but scoped to the peptide mask (receptor positions don't care,
        but we also don't want their random noise to pollute the mean).

        pep_mask: (B,L) bool; trans_noise: (B,L,3). Returns (B,L,3) with the
        peptide-region mean subtracted from all peptide positions. Non-peptide
        positions are returned unchanged (they will be overwritten by receptor
        coords downstream anyway, but we keep the tensor shape clean).
        """
        mask_f = pep_mask.to(trans_noise.dtype).unsqueeze(-1)            # (B,L,1)
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)            # (B,1,1)
        mean = (trans_noise * mask_f).sum(dim=1, keepdim=True) / denom    # (B,1,3)
        return trans_noise - mean * mask_f

    def corrupt_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create interpolants and endpoints for PH-Flow with optional minibatch OT alignment."""
        aa_1 = batch['aa']
        # Apply centralized coord scaling once (idempotent: callers must pass raw angstroms).
        pos = batch['pos_heavyatom'] * self.coord_scale
        gen_mask = batch['generate_mask'].bool()
        res_mask = batch['res_mask'].bool()

        rot_1 = construct_3d_basis(pos[:, :, BBHeavyAtom.CA], pos[:, :, BBHeavyAtom.C], pos[:, :, BBHeavyAtom.N])
        trans_1 = pos[:, :, BBHeavyAtom.CA]
        ang_1 = batch['torsion_angle']

        B, L = aa_1.shape
        device = aa_1.device

        t = torch.rand((B, 1), device=device)
        # avoid t extremely close to 1 to prevent tiny dt -> large 1/dt amplification
        t = t * 0.9 + 0.05

        # noise
        trans_0 = torch.randn((B, L, 3), device=device) * self.flow_cfg.sigma_trans
        rot_0 = so3.quat_to_rotmat(self._random_quat((B, L), device=device, dtype=trans_1.dtype))
        ang_0 = torus.tor_random_uniform(B, L, ang_1.shape[-1], device=device, dtype=ang_1.dtype)

        pep_mask = gen_mask & res_mask
        # Zero-mean noise on peptide residues (PFF-style _centered_gaussian). Done
        # before OT: permutation is mean-invariant, so either order works, but
        # centering first keeps the pairwise_cost diagonal tight.
        if pep_mask.any():
            trans_0 = self._center_noise_on_pep(trans_0, pep_mask)
        if self.flow_cfg.use_ot and pep_mask.any() and self.flow_cfg.ot_mode == "minibatch":
            # Mini-ROT (paper Eq. 13): one assignment between the B priors and the B data samples.
            trans_0, rot_0, ang_0 = minibatch_ot.minibatch_ot_pair(
                (trans_0, rot_0, ang_0), (trans_1, rot_1, ang_1), pep_mask
            )
        elif self.flow_cfg.use_ot and pep_mask.any():
            # Legacy ot_mode='residue': OT aligns noisy samples to data within peptide residues, improving
            # straightness of flow lines. We build per-b cost matrices first, then
            # pad to uniform N_max and issue ONE batched solve_ot call — this lets
            # the internal ThreadPoolExecutor (src/data/minibatch_ot.py) actually
            # parallelize Hungarian across B. The prior per-b `solve_ot` loop
            # always hit the `B <= 2` inline branch inside solve_ot and wasted R6.
            n_per_b = pep_mask.sum(dim=1).tolist()  # single CPU sync, unavoidable
            n_max = max(n_per_b) if n_per_b else 0
            if n_max > 0:
                idxs: list = [None] * B
                srcs: list = [None] * B  # (trans, rot, ang) snapshots pre-shuffle
                # Padding cost uses `1e6` + zero diagonal so Hungarian pins padding
                # rows to padding cols (trivial assignment, no interference with
                # real peptide rows). Matches the `mask1` sentinel in pairwise_cost.
                cost_batch = torch.full(
                    (B, n_max, n_max), 1e6, device=device, dtype=trans_1.dtype
                )
                diag_arange = torch.arange(n_max, device=device)
                for b in range(B):
                    n = n_per_b[b]
                    if n == 0:
                        cost_batch[b, diag_arange, diag_arange] = 0.0
                        continue
                    idx = torch.nonzero(pep_mask[b], as_tuple=False).squeeze(-1)
                    z0_b = (trans_0[b:b + 1, idx], rot_0[b:b + 1, idx], ang_0[b:b + 1, idx])
                    z1_b = (trans_1[b:b + 1, idx], rot_1[b:b + 1, idx], ang_1[b:b + 1, idx])
                    cost_batch[b, :n, :n] = minibatch_ot.pairwise_cost(z0_b, z1_b)[0]
                    if n < n_max:
                        pad_rows = diag_arange[n:]
                        cost_batch[b, pad_rows, pad_rows] = 0.0
                    idxs[b] = idx
                    # Snapshot pre-shuffle so the scatter below doesn't read what it just wrote.
                    srcs[b] = (trans_0[b, idx].clone(), rot_0[b, idx].clone(), ang_0[b, idx].clone())

                # Single batched Hungarian. For B>2 the ThreadPoolExecutor in
                # solve_ot dispatches scipy calls across threads; for B<=2 this
                # degenerates to inline solves (identical to pre-refactor behavior).
                perm_batch = minibatch_ot.solve_ot(cost_batch)  # (B, n_max)

                for b in range(B):
                    if idxs[b] is None:
                        continue
                    n = n_per_b[b]
                    perm = perm_batch[b, :n]  # values in [0, n) by padding construction
                    idx = idxs[b]
                    src_trans, src_rot, src_ang = srcs[b]
                    trans_0[b, idx] = src_trans[perm]
                    rot_0[b, idx] = src_rot[perm]
                    ang_0[b, idx] = src_ang[perm]

        
        sigma_min = getattr(self.flow_cfg, 'sigma_min', 1e-4)
        trans_t = (1.0 - t[..., None]) * trans_0 + t[..., None] * trans_1
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

        # pos_emb has max_len+8 entries. Fail loudly if the batch exceeds that:
        # silently clamping the overflow collapses two distinct residues onto the same
        # positional embedding, which corrupts the whole sequence without any error.
        pos_capacity = self.max_len + 8
        if L > pos_capacity:
            raise ValueError(
                f"Sequence length L={L} exceeds positional embedding capacity "
                f"(max_len + 8 = {pos_capacity}). Increase `model.max_len` in the config."
            )

        # positional ids: clamp into max_len
        pos_ids = batch.get('res_nb', torch.arange(L, device=seq_t.device)[None].repeat(B, 1))
        if isinstance(pos_ids, torch.Tensor):
            pos_ids = _squeeze_last_if_one(pos_ids, dims=[-1], name='res_nb')
        # res_nb is author-supplied (PDB numbering, can be >> L); clamping into
        # [0, pos_capacity-1] is fine here — the L-level overflow check above is
        # what catches the real "sequence too long" bug.
        pos_ids = pos_ids.clamp(min=0, max=pos_capacity - 1)
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
                esm_cond, _ = self.esm_conditioner(self._esm_tokens(seq_t, gen_mask), res_mask)
        if esm_cond is not None:
            if not isinstance(esm_cond, torch.Tensor):
                raise TypeError(f"esm_cond must be a torch.Tensor, got {type(esm_cond)}")
            if esm_cond.ndim == 4 and esm_cond.shape[-2] == 1:
                esm_cond = esm_cond.squeeze(-2)
            if esm_cond.shape[:2] != (B, L):
                raise ValueError(f"esm_cond must have shape (B,L,*) matching the sequence length, got {tuple(esm_cond.shape)} vs (B={B},L={L})")
            # === 先用esm_proj投影到d_model，再送入esm_adapter（LoRA�?===
            if self.esm_proj is not None:
                esm_cond = self.esm_proj(esm_cond)  # [B, L, hidden_size] -> [B, L, d_model]
            if self.esm_adapter is not None:
                esm_cond = self.esm_adapter(esm_cond, chirality_token=chir)  # [B, L, d_model] -> [B, L, d_model]
            if self.trunk_type == "geometric":
                # receptor context only (peptide tokens carry no pLM semantics in the paper's design)
                esm_cond = esm_cond * ((gen_mask == 0) & res_mask).unsqueeze(-1).to(esm_cond.dtype)
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

        # Chiral gate G(c) = c * sigmoid(gamma): sigmoid(gamma) lives inside ChiralBridgeLoRA, c is supplied here.
        # The gate scope wraps *every* trunk path (the old AFI path forgot it, leaving LoRA permanently off).
        rot6d_input = rotmat_to_6d(rot_t)
        key_pad_mask = self._safe_padding_mask(res_mask)
        if self.chi_lora_cfg.enabled:
            gate = chir.view(B, 1, 1).to(device=seq_t.device, dtype=x.dtype)
            if self.trunk_type == "joint":
                gate = gate * (gen_mask.bool() & res_mask.bool()).unsqueeze(-1).to(x.dtype)  # legacy: peptide tokens only
        else:
            gate = None

        use_afi = self.afi_cfg.enabled and len(self.afi_modules) > 0
        with self._chi_lora_gate(gate) if gate is not None else contextlib.nullcontext():
            if self.trunk_type == "geometric":
                hook = (lambda i, h_in: self.afi_modules[i](h_in, trans_t, rot6d_input)) if use_afi else None
                h = self.trunk(x, trans_t / self.coord_scale, res_mask, gen_mask.bool(), pre_layer_hook=hook)
            elif use_afi:
                h = x
                for i, layer in enumerate(self.trunk.layers):
                    h = self.afi_modules[i](h, trans_t, rot6d_input)
                    h = layer(h, src_key_padding_mask=key_pad_mask)
            else:
                h = self.trunk(x, src_key_padding_mask=key_pad_mask)

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

    # Kendall-style uncertainty weighting bounds: σ² ∈ [e^-5, e^5] ≈ [6.7e-3, 148]
    # keeps precisions = exp(-log_var) in [e^-5, e^5] and prevents runaway during AMP.
    _LOG_VAR_MIN = -5.0
    _LOG_VAR_MAX = 5.0

    def _compute_total_loss(self, cfm_loss, sym_loss, seq_loss, sym_weight_override):
        """Combine component losses. Returns (total, w_cfm, w_sym, w_seq) for logging.

        Modes:
          - Manual (loss_log_vars is None): honor cfm_weight/seq_weight and either the
            override or flow_cfg.sym_weight.
          - Auto Kendall (loss_log_vars present): L = 0.5*exp(-s)*L_i + 0.5*s. When
            sym_weight_override is provided (sym warmup still active), sym keeps manual
            weight while cfm/seq are learned.
        """
        if self.loss_log_vars is None:
            w_cfm = self.flow_cfg.cfm_weight
            w_seq = self.flow_cfg.seq_weight
            w_sym = sym_weight_override if sym_weight_override is not None else self.flow_cfg.sym_weight
            total = w_cfm * cfm_loss + w_sym * sym_loss + w_seq * seq_loss
            return total, w_cfm, w_sym, w_seq

        log_vars = self.loss_log_vars.clamp(min=self._LOG_VAR_MIN, max=self._LOG_VAR_MAX)
        prec = torch.exp(-log_vars)
        w_cfm = 0.5 * prec[0]
        w_seq = 0.5 * prec[2]
        L_cfm = w_cfm * cfm_loss + 0.5 * log_vars[0]
        L_seq = w_seq * seq_loss + 0.5 * log_vars[2]

        if sym_weight_override is not None:
            w_sym = sym_weight_override
            total = L_cfm + w_sym * sym_loss + L_seq
        else:
            w_sym = 0.5 * prec[1]
            L_sym = w_sym * sym_loss + 0.5 * log_vars[1]
            total = L_cfm + L_sym + L_seq
        return total, w_cfm, w_sym, w_seq

    def forward(self, batch: Dict[str, torch.Tensor], sym_weight_override: Optional[float] = None) -> Dict[str, torch.Tensor]:
        t, seq_t, trans_t, rot_t, ang_t, aa_1, trans_1, rot_1, ang_1, trans_0, rot_0, ang_0 = self.corrupt_batch(batch)

        # Cache raw ESM conditioner output once and share across main/sym branches.
        # ESM features depend only on the sequence (seq_t, res_mask) — parity flips positions,
        # not amino acids — so the mirror branch must see the exact same ESM tensor.
        # predict_x1 still runs the cheap per-branch esm_proj + chirality-gated esm_adapter on top.
        if self.esm_conditioner is not None and batch.get('esm_cond', None) is None:
            res_mask_for_esm = batch['res_mask'].bool()
            with torch.no_grad():
                esm_cond_raw, _ = self.esm_conditioner(self._esm_tokens(seq_t, batch['generate_mask']), res_mask_for_esm)
            batch['esm_cond'] = esm_cond_raw

        out = self.predict_x1(batch, t, seq_t, trans_t, rot_t, ang_t)

        res_mask = batch['res_mask'].bool()
        gen_mask = batch['generate_mask'].bool()
        mask = res_mask & gen_mask

        pred_states = (out['pred_trans_1'], out['pred_rot_1'], out['pred_ang_1'])
        target_states = (trans_1, rot_1, ang_1)

        sigma_min = getattr(self.flow_cfg, 'sigma_min', 1e-4)
        cfm_loss = ph_flow.flow_matching_loss(
            pred_states, target_states,
            trans_t, rot_t, ang_t, t,
            trans_0, rot_0, ang_0,
            trans_1, rot_1, ang_1,
            mask,
            weights=(self.flow_cfg.cfm_w_trans, self.flow_cfg.cfm_w_rot, self.flow_cfg.cfm_w_ang),
            sigma_min=sigma_min,
            loss_type=self.flow_cfg.loss_type,
        )

        # Skip symmetry loss computation entirely if weight is zero to avoid
        # autograd graph pollution from the second predict_x1 call.
        effective_sym_weight = sym_weight_override if sym_weight_override is not None else self.flow_cfg.sym_weight
        if effective_sym_weight > 0:
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
                sigma_min=sigma_min,
                loss_type=self.flow_cfg.loss_type,
            )
        else:
            sym_loss = torch.tensor(0.0, device=cfm_loss.device)

        # CE only on peptide residues
        if mask.any():
            seq_loss = F.cross_entropy(out['pred_seq_logits'][mask], aa_1[mask].clamp(max=self.aatype_vocab - 1))
        else:
            seq_loss = torch.tensor(0.0, device=aa_1.device)

        total, eff_w_cfm, eff_w_sym, eff_w_seq = self._compute_total_loss(
            cfm_loss, sym_loss, seq_loss, sym_weight_override
        )

        # -------- Diagnostic accuracy / error metrics --------
        # Computed under no_grad to avoid autograd graph extension; these are
        # logged to TensorBoard for interpretability but do NOT affect the loss
        # or gradients. Physical units: trans MAE in angstroms (coord_scale
        # reversed), rot/torsion error in radians (outer logger converts to deg).
        with torch.no_grad():
            zero = torch.zeros((), device=aa_1.device)
            if mask.any():
                pred_aa = out['pred_seq_logits'].argmax(dim=-1)
                aa_target = aa_1.clamp(max=self.aatype_vocab - 1)
                aa_accuracy = (pred_aa[mask] == aa_target[mask]).float().mean()

                # trans_1 is already coord_scale-scaled by corrupt_batch; undo
                # to report in raw angstroms.
                trans_mae_scaled = (out['pred_trans_1'] - trans_1)[mask].abs().mean()
                trans_mae_ang = trans_mae_scaled / max(self.coord_scale, 1e-8)

                rot_err = so3.geodesic_distance(out['pred_rot_1'], rot_1)  # (B, L)
                rot_err_rad = rot_err[mask].mean()

                ang_diff = torus.tor_logmap(out['pred_ang_1'], ang_1)
                if 'torsion_angle_mask' in batch:
                    ang_mask_full = mask.unsqueeze(-1) & batch['torsion_angle_mask'].bool()
                else:
                    ang_mask_full = mask.unsqueeze(-1).expand_as(ang_diff)
                if ang_mask_full.any():
                    torsion_err_rad = ang_diff.abs()[ang_mask_full].mean()
                else:
                    torsion_err_rad = zero
            else:
                aa_accuracy = zero
                trans_mae_ang = zero
                rot_err_rad = zero
                torsion_err_rad = zero

        return {
            'loss': total,
            'cfm_loss': cfm_loss,
            'sym_loss': sym_loss,
            'seq_loss': seq_loss,
            'w_cfm': torch.as_tensor(eff_w_cfm),
            'w_sym': torch.as_tensor(eff_w_sym),
            'w_seq': torch.as_tensor(eff_w_seq),
            # Diagnostic-only (no grad, no impact on training)
            'aa_accuracy': aa_accuracy,
            'trans_mae_ang': trans_mae_ang,
            'rot_err_rad': rot_err_rad,
            'torsion_err_rad': torsion_err_rad,
        }

    @torch.no_grad()
    def decode_to_heavyatom15(self, aa: torch.Tensor, trans: torch.Tensor, rotmats: torch.Tensor, angles: torch.Tensor, chirality: torch.Tensor, peptide_mask: torch.Tensor, hard_constraints: bool = True):
        """Geometric decoder with optional hard chirality enforcement.

        Delegates to HardGeometricDecoder so the identity
            Mirror(L_geom(theta)) = D_geom(-theta)
        (paper Eq. (10)) is applied consistently: for residues with chirality=1 in the
        peptide mask, we both mirror the backbone frame (diag(1,1,-1)) and negate the
        torsion angles before feeding them into the standard L-amino-acid reconstruction.

        When hard_constraints=False we skip the mirror + angle flip (used by ablations
        that want to measure the raw model's chiral error).
        """
        if hard_constraints:
            from src.models.hard_geometric_decoder import HardGeometricDecoder
            decoder = HardGeometricDecoder()
            return decoder(aa=aa, trans=trans, rotmats=rotmats, angles=angles,
                           chirality=chirality, peptide_mask=peptide_mask)

        # Ablation path: no chirality enforcement at all.
        pos14, _, _ = full_atom_reconstruction(R_bb=rotmats, t_bb=trans, angles=angles, aa=aa)
        pos15 = F.pad(pos14, pad=(0, 0, 0, 15 - 14), value=0.0)
        mask15 = get_heavyatom_mask(aa)
        return pos15, mask15

    @torch.no_grad()
    def sample(self, batch: Dict[str, torch.Tensor], num_steps: int = 30, angle_purify: bool = True):
        device = batch['aa'].device
        aa_gt = batch['aa']
        # Centralized coord scaling (same units as corrupt_batch); decoded outputs stay in
        # scaled units -- callers/scripts are responsible for unscaling via self.coord_scale.
        pos = batch['pos_heavyatom'] * self.coord_scale
        gen_mask = batch['generate_mask'].bool()
        res_mask = batch['res_mask'].bool()
        B, L = aa_gt.shape

        # fixed context (receptor)
        rot_1 = construct_3d_basis(pos[:, :, BBHeavyAtom.CA], pos[:, :, BBHeavyAtom.C], pos[:, :, BBHeavyAtom.N])
        trans_1 = pos[:, :, BBHeavyAtom.CA]
        ang_1 = batch['torsion_angle']

        # init state at t=0
        # Same zero-mean noise as corrupt_batch: sampling from an un-centered prior
        # would start the flow from a structure whose CoM is translated relative to
        # the receptor, and the model (trained on centered noise) has never seen
        # that input distribution.
        pep_for_init = gen_mask & res_mask
        init_trans_noise = torch.randn((B, L, 3), device=device) * self.flow_cfg.sigma_trans
        if pep_for_init.any():
            init_trans_noise = self._center_noise_on_pep(init_trans_noise, pep_for_init)
        trans = torch.where(gen_mask[..., None], init_trans_noise, trans_1)
        rot = torch.where(gen_mask[..., None, None], so3.quat_to_rotmat(self._random_quat((B, L), device=device, dtype=trans.dtype)), rot_1)
        ang = torch.where(gen_mask[..., None], torus.tor_random_uniform(B, L, ang_1.shape[-1], device=device, dtype=ang_1.dtype), ang_1)
        seq = torch.where(gen_mask, torch.full_like(aa_gt, du.MASK_TOKEN_INDEX), aa_gt)
        seq = torch.where(res_mask, seq, torch.full_like(seq, du.MASK_TOKEN_INDEX))

        # t-schedule now lives in the training distribution [min_t, 1]. Old code
        # started at t_k=0, which is outside corrupt_batch's training range
        # (t ∈ [0.05, 0.95]); the very first predict_x1 call was therefore an OOD
        # extrapolation and the trajectory started off the learned manifold.
        # PFF/dflow both use linspace(min_t, 1, num_steps+1). We reproduce that.
        min_t = float(getattr(self.flow_cfg, 'sample_min_t', 0.05))
        dt = (1.0 - min_t) / float(num_steps)
        for k in range(num_steps):
            t_k = min_t + dt * k
            t = torch.full((B, 1), t_k, device=device)
            pred = self.predict_x1(batch, t, seq, trans, rot, ang)

            pep = gen_mask & res_mask
            if pep.any():
                remaining = 1.0 - t_k
                # last step: alpha = 1 (land exactly on prediction); otherwise alpha = dt / (1 - t_k).
                alpha = 1.0 if k == num_steps - 1 else min(1.0, dt / max(remaining, 1e-6))
                alpha_t = torch.tensor(alpha, device=device, dtype=trans.dtype)

                trans_new = trans + alpha_t * (pred['pred_trans_1'] - trans)
                rot_new = so3.geodesic_t(alpha_t, pred['pred_rot_1'], rot)
                ang_new = torus.tor_geodesic_t(alpha_t, pred['pred_ang_1'], ang)

                trans = torch.where(pep[..., None], trans_new, trans)
                rot = torch.where(pep[..., None, None], rot_new, rot)
                ang = torch.where(pep[..., None], ang_new, ang)

                # Sequence update follows the training MASK-schedule: at time t_k
                # the training corruption keeps a fraction (1-t_k) of peptide
                # residues MASK'd, so we only "commit" predicted residues where
                # rand < t_k, leaving the rest as MASK for the next forward to
                # re-predict. The old code argmax-committed every position every
                # step, which (a) forced the model to condition on hard one-hot
                # tokens it never saw during training, and (b) let low-confidence
                # early predictions lock the rest of the trajectory. Final step
                # commits everything so the returned seq is fully decoded.
                seq_pred = torch.argmax(pred['pred_seq_logits'], dim=-1)
                if k == num_steps - 1:
                    seq = torch.where(pep, seq_pred, seq)
                else:
                    commit = (torch.rand(pep.shape, device=device) < t_k) & pep
                    seq = torch.where(commit, seq_pred, seq)

            if angle_purify:
                ang = ang % (2 * math.pi)

        # Unscale translations back to the caller's unit system (raw angstroms).
        # Rotations / angles are scale-invariant and stay as-is. After this line,
        # all outputs live in the same unit system as the original batch['pos_heavyatom'],
        # so downstream decode/merge/save steps can treat them as plain angstroms.
        trans = trans / self.coord_scale

        return {
            'seq': seq,
            'trans': trans,
            'rotmats': rot,
            'angles': ang,
        }


# Backward-compatible alias: the class was previously named UniChiFlowModel.
UniChiFlowModel = PDexFlowModel
