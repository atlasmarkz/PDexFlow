# UniChiFlow 代码修复与重构方案

**范围**：2026-05-08 对话中提出并（大部分）落地的代码修复与重构。
**核心约束**：全程**不改变理论逻辑** —— PH-Flow 奇偶等变性、Hard Decoder 的 `Mirror(L_geom(θ)) = D_geom(-θ)` 恒等式、ChiLoRA chirality gating、Minibatch OT 直线轨迹、SE(3)×T^k flow matching 目标均未改动。

---

## 一、问题分类

- **🔴 R 类**：correctness / 数值稳定性 / AMP 安全 → 直接影响训练正确性，必须做
- **🟡 Y 类**：工程质量（fail-loud、命名、解析健壮性、并行、fp32 包装）→ 影响可维护性与可诊断性

---

## 二、已完成修改清单

### 🔴 R 类（全部完成，8/8）

| ID | 标题 | 核心改动 | 主要文件 |
|---|---|---|---|
| **R1** | 重写 loss 加权分支 + log_vars clamp | Kendall `_LOG_VAR_MIN=-5, _LOG_VAR_MAX=5`，precision `= exp(-log_var)` 有界；`_compute_total_loss` 独立方法替换原 forward 里的嵌套 if/elif/else | `src/models/chiralformer.py` |
| **R2** | SE(3) / torus 运算强制 fp32 | `_fp32` 装饰器（`torch.autocast(enabled=False)` + `.float()` + 最终 dtype 还原），包住 `rotmat_to_quat/quat_to_rotmat/slerp/geodesic_t/exp_map/log_map/geodesic_distance/tor_expmap/tor_logmap/tor_geodesic_t` | `src/data/so3.py`, `src/data/torus.py` |
| **R3** | sample loop 改 x1-predict 等价形式 | `alpha = 1.0 if last_step else min(1.0, (1/num_steps)/max(remaining, 1e-6))`，配合 `so3.geodesic_t` / `torus.tor_geodesic_t`；避免 `(pred_x1 - x_t)/(1-t)` 在 t→1 时炸 | `src/models/chiralformer.py:sample()` |
| **R4** | 坐标归一化集中化 | `UniChiFlowModel.coord_scale=0.1`，`corrupt_batch` 和 `sample` 入口缩放，`sample` 返回前反缩放；三个脚本移除 `* 0.1` / `/ 0.1` | `chiralformer.py`, 3 个训练/推理脚本 |
| **R5** | v_clip 改到 delta 上 | `flow_matching_loss` / `symmetry_consistency_loss` 先 clamp x1-空间位移（`cap = v_clip * dt`），再除 dt；fp16 下中间量保持有限，数学等价 `|v| ≤ v_clip` | `src/flow.py` |
| **R6** | OT 批量并行 | `ThreadPoolExecutor`（scipy 释放 GIL）并行 B 个 Hungarian；B≤2 走内联；池复用 | `src/data/minibatch_ot.py` |
| **R7** | sym 支路共享 ESM + 浅拷贝 | `forward()` 先算一次 raw ESM cond 塞进 `batch['esm_cond']`；`_clone_batch_with_chirality` 改 `dict(batch)` 浅拷贝，只替换 chirality | `chiralformer.py`, `flow.py` |
| **R8** | 统一 Hard Decoder 角度处理 | `decode_to_heavyatom15` 委托到 `HardGeometricDecoder`，不再在外部对 D-peptide 做二次翻转 | `chiralformer.py` |

### 🟡 Y 类（10/10 完成）

| ID | 标题 | 核心改动 | 主要文件 |
|---|---|---|---|
| **Y1** | fail-loud NaN 处理 | 两个训练脚本 `_sanitize_gradients`（静默 `nan_to_num_`）→ `_check_and_clip_gradients`，检测到 NaN/Inf 直接 skip opt step + bad_step 计数 + LR decay | `train_unichiflow.py`, `train_lora_real_d.py` |
| **Y2** | early stopping 改名 + 澄清 | `EarlyStopping` → `BestCheckpointKeeper`（实际作用是保存最佳 ckpt，stop 分支几乎不触发），docstring/日志澄清 | `train_unichiflow.py` |
| **Y3** | Trainer 抽象（Step 2 + Step 3）| Step 2 工厂 `src/experiments/components.py`；Step 3 `do_train_step` 纯函数 `src/experiments/train_step.py`。Backbone 主循环从 ~80 行嵌套 if 缩到 ~50 行 dispatch。CPU smoke (20 步 ~3 秒) 通过 bit-level 回归验证。LoRA 循环形态不同（无 AMP/grad_accum/LR decay），留给后续 Step 5 统一 | `components.py`, `train_step.py`, `train_unichiflow.py` |
| **Y4** | 超长序列 assert | `predict_x1` 入口 `L > max_len + 8` 显式 raise，替换原本静默 clamp（会把不同残基映射到同一位置嵌入） | `chiralformer.py` |
| **Y5** | Transformer 空序列保护 | `_safe_padding_mask` 静态方法：某 batch row 全为 padding 时，把 pos 0 翻成 valid，避免 `nn.TransformerEncoder` 全 mask 产 NaN | `chiralformer.py` |
| **Y6** | ESM2 fp32 外壳 | `ESM2Conditioner.forward` 套 `torch.autocast(enabled=False)`，pretrained 权重在 fp32 下跑，避开 fp16 softmax 溢出 | `src/models/semantic_conditioner.py` |
| **Y7** | torus logmap 换 atan2 形式 | 已经是 atan2 实现（`tor_logmap = atan2(sin(Δ), cos(Δ))`），确认无需再改 | `src/data/torus.py` |
| **Y8** | OT fallback 修复 | scipy 缺失时的 `_greedy_permutation`：逐行贪婪，已选列置 `+inf`，保证输出是合法排列（原本 `argmin` 会出现重复列 → gather 错乱） | `src/data/minibatch_ot.py` |
| **Y9** | ChiLoRA gate buffer 时序修复 | `_set_chi_lora_gate` 加 assert + `_chi_lora_gate` 上下文管理器；预防 sym/main 支路串用 gate buffer | `chiralformer.py` |
| **Y10** | ChiLoRAConfig 解析显式化 | 显式接受 `Mapping / DictConfig / ChiLoRAConfig`；类型不对 `TypeError`，不再吞异常回落到 disabled | `chiralformer.py` |

### 🔵 C 类（config / 数据配合修复）

| ID | 标题 | 核心改动 | 主要文件 |
|---|---|---|---|
| **C1** | `model.max_len` 512 → 1024 | 真实 PepMerge 数据 receptor 常见 100–500 残基，512 只覆盖 87.5%、12.5% 样本训练时触发 Y4 assert 崩溃。实测分布：p90=688, p95=768, p99=1001, max=1274。1024 覆盖 99.4%。参数开销:每 config 下游 +65K–131K（<8%）。所有 8 个 yaml 同步更新保持 checkpoint 加载维度一致 | `configs/unichiflow_backbone*.yaml`, `configs/unichiflow_4060ti.yaml`, `configs/unichiflow_inference*.yaml`, `configs/unichiflow_eval*.yaml`, `configs/unichiflow_lora.example.yaml` |

### 🟢 P 类（perf / 工程加速）

| ID | 标题 | 核心改动 | 主要文件 | 预期收益 |
|---|---|---|---|---|
| **P1** | OT 批量化 + 利用 R6 线程池 | `corrupt_batch` 里的 `for b in range(B): solve_ot(cost)` 改为：先收集所有 per-b `pairwise_cost` 结果,padding 到 `(B, N_max, N_max)`(padding 行用对角=0 + off-diag=1e6 钉死,不抢真实列),再**一次 `solve_ot`**,内部 `ThreadPoolExecutor`(B>2 时)并行 Hungarian。原代码每次 `solve_ot(cost)` 里 B'=1 永远走 inline 路径,R6 的 threadpool 白加了。CPU smoke bs=2 bit-level 回归通过 | `src/models/chiralformer.py:corrupt_batch` | GPU bs=16 下 OT 步骤 2–5x,总 step 10–20% |

### 📊 诊断指标（不改训练,只是可观测性）

| ID | 指标 | 单位 | 含义 | 实现 |
|---|---|---|---|---|
| **M1** | `train/skip_rate` | 0–1 | 累计 skip 率 = `total_skips / step`。<2% 健康,>10% 停机排查 | `train_unichiflow.py` 外层累计 |
| **M2** | `train/bad_step_streak` | int | 当前连续 skip 数(bad_step 镜像),每 ok step 清零。长期 ≥5 = 系统性问题 | `train_unichiflow.py` 外层累计 |
| **M3** | `train/aa_accuracy` | % | 序列恢复率 AAR,argmax(pred_aa) == aa_1 的比例。随机基线 5%(20 类),训到 ~40% 说明序列能学 | `chiralformer.forward` 返回 |
| **M4** | `train/trans_mae_ang` | Å | 平移 MAE,`coord_scale` 已反算回真实埃单位。<2 Å 算不错 | `chiralformer.forward` 返回 |
| **M5** | `train/rot_err_deg` | ° | SO(3) 测地角误差(radians→deg 在外层转换)。<30° 算不错 | `chiralformer.forward` 返回 |
| **M6** | `train/torsion_err_deg` | ° | 扭转角环面距离(atan2 wrapping)。<45° 算不错 | `chiralformer.forward` 返回 |

指标 M3–M6 全部在 `torch.no_grad()` 下计算,**不影响训练数值**(bit-level 回归验证)、不影响速度(实测 CPU smoke 差 3% 在测量噪声内)。

---

## 三、Trainer 抽象（Y3）五步 Roadmap

| 步骤 | 内容 | 状态 |
|---|---|---|
| **Step 1** | 固定基线（跑 backbone ~1000 步 + LoRA ~500 步，记录 loss/grad/skip 曲线） | ⏳ 待做（需 GPU 机器）。本机 CPU smoke (~20 步) 已跑通，作为 Step 3 回归参照 |
| **Step 2** | 抽纯函数工厂（dataset/dataloader/model/optimizer/ckpt-load） | ✅ **已完成** |
| **Step 3** | 把 train step 抽成纯函数 `do_train_step(batch, model, optim, scaler, cfg, step, sym_weight_override) -> StepOutcome`（AMP / grad_accum / NaN skip 全塞进去；LR decay 留在外层）| ✅ **已完成**（backbone，CPU smoke bit-level 回归通过） |
| **Step 4** | 统一 config schema（显式化 `use_amp / grad_accum_steps / sym_warmup_* / freeze_backbone / ckpt_path` 等） | ⏳ 待做 |
| **Step 5** | 抽 `Trainer` 基类，2–3 个 hook（`build_dataset / on_log / on_save`）；`BackboneTrainer` 和 `LoRATrainer` 作为子类。Step 5 也承担把 LoRA 对齐到 `do_train_step`（当前 LoRA 循环无 AMP/grad_accum/LR decay，形态不同，故意留在 Step 3 之外避免引入行为变更）| ⏳ 待做 |

### Step 2 已完成的具体改动

**新增文件** `src/experiments/components.py`（138 行），提供：

| 工厂函数 | 说明 |
|---|---|
| `build_dataset(cfg, kind)` | `kind="pepmerge"` → PepMergeSubsetDataset/TwinUniversePepDataset；`kind="real_d"` → RealDPepDataset |
| `build_dataloader(ds, cfg, eight=True)` | `DataLoader + PaddingCollate`，一处定义 |
| `inf_iterator(loader)` | 替换原先两处重复的 `_inf_iterator` |
| `build_model(cfg, include_multitask_weights)` | `UniChiFlowModel` 构造；`include_multitask_weights` 保留 backbone/LoRA 既有的不对称（backbone 传 `cfm_weight/sym_weight`，LoRA 不传） |
| `build_optimizer(model, cfg, trainable_only)` | `trainable_only=True` 走 requires_grad filter（LoRA），`False` 走全参（backbone） |
| `load_initial_weights(model, cfg, device)` | `load_backbone_into_lora` 的统一入口，无 `sample.ckpt_path` 时打 warning + 返回原 model |

**调用方改造**：
- `train_unichiflow.py`：40+ 行内联构造 → 5 行工厂调用；删除 `DataLoader / PaddingCollate / TwinUniversePepDataset / PepMergeSubsetDataset / UniChiFlowModel` 直接导入
- `train_lora_real_d.py`：45+ 行内联构造 → 5 行工厂调用；删除 `DataLoader / PaddingCollate / RealDPepDataset / UniChiFlowModel / load_backbone_into_lora` 直接导入

**保真点**（故意保留的既有不一致，待 Step 4 处理）：
- Backbone 显式传 `cfm_weight/sym_weight`，LoRA 走模型默认值
- `kind` 由调用方显式传字符串，避免引入新的 config 字段
- LoRA 的 `trainable_params` 列表仍在脚本中计算（被下游 `_check_and_clip_gradients` / `clip_grad_norm_` 复用）

### Step 3 已完成的具体改动

**新增文件** `src/experiments/train_step.py`（~150 行），提供：

| 导出 | 说明 |
|---|---|
| `StepKind` | `Literal["ok", "skip_loss_nonfinite", "skip_grad_nonfinite", "accum"]` |
| `StepConfig` | frozen dataclass：`use_amp / amp_device / grad_accum_steps / max_grad_norm / max_grad_value` |
| `StepOutcome` | dataclass：`kind / loss / components / grad_norm` —— 外层据此 dispatch |
| `do_train_step(*, batch, model, optim, scaler, cfg, step, sym_weight_override) -> StepOutcome` | 一次 forward + backward + （accum 边界时）opt step |

**调用方改造**：
- `train_unichiflow.py` 主循环从 `~80` 行嵌套 if/elif/continue 缩到 `~50` 行纯 dispatch
- 删除 `clip_grad_norm_` 直接导入和 `_check_and_clip_gradients` 内嵌定义（现在都在 `train_step.py`）
- 外层保留的状态：`smooth_loss / global_step / bad_step / _decay_lr / sym_warmup` —— 这些是跨 step 的，不属于 `do_train_step` 职责

**LR decay 设计决定**：让外层看 `outcome.kind`，自己计 `bad_step`，自己决定什么时候 `_decay_lr()`。这样 `do_train_step` 保持无状态，单步可单测；LoRA 后续要接入时可以直接复用而不用继承 LR decay 策略。

**回归保证**：
- CPU smoke（20 步，bs=2，subset=0.01，seed=2024）refactor 前后 loss/grad/skip trace **bit-identical**
- 保留了 autocast / scaler.unscale_ / clip_grad_norm_ / scaler.step 的调用顺序
- `pbar.set_description` 字符串与 TensorBoard scalar key 原样保留

**保真点**（故意不动）：
- LoRA 训练脚本未迁移：循环形态不同（无 AMP、无 grad_accum、无 LR decay、`trainable_params` 子集），强行塞进 `do_train_step` 会引入行为变更；等 Step 5 统一 Trainer 抽象时再处理

---

## 四、待完成的测试

### 1. Backbone 基线（Step 3 之前必须跑）

**前置**：GPU 机器 + `data/pepmerge/` 数据集 + `.cache/pep_cache_subset`

**冒烟（~1 分钟，先跑这个）**：
```bash
python -m src.experiments.train_unichiflow train.max_iters=5 train.save_freq=1000
```
预期：能进入 tqdm 首个 step，无 `TypeError: UniChiFlowModel()` / `ImportError`。如果 hydra 启动后立刻报错，是工厂参数跟 `UniChiFlowModel.__init__` 不匹配 —— 贴错误回来。

**基线（~1000 步）**：
```bash
mkdir -p baseline_logs
python -m src.experiments.train_unichiflow \
    train.max_iters=1000 train.save_freq=500 train.seed=2024 \
    2>&1 | tee baseline_logs/backbone_step2.log
```

**产物**（**整个目录留存**，Step 3 / 5 完成后重跑并 diff）：
- `outputs/<timestamp>/` —— TensorBoard：`loss / smooth_loss / cfm_loss / sym_loss / seq_loss / grad_norm / lr / skip_rate / bad_step_streak`
- `outputs/<timestamp>/checkpoints/` —— `best_model.pt` + `500.pt` + `1000.pt`
- `baseline_logs/backbone_step2.log` —— tqdm 流（含 skip 事件、LR decay 事件）

**训练健康度诊断**（在 TensorBoard 里同时看这两条曲线）：
- `train/skip_rate`：累计 skip 率(总 skip 次数 / 累计 step)。**<2% 健康**,2–10% 需关注 LR / 数据,>10% 停机排查
- `train/bad_step_streak`：当前**连续** skip 次数。触发 LR decay 的阈值是 10,如果峰值长期 ≥5 说明某批 batch 持续有问题(而不是偶发离群)。Smoke 下 3 次 skip 都是孤立的(峰值=1),属于偶发型

### 2. LoRA 基线（依赖 backbone 基线产物）

**前置**：
- `configs/unichiflow_lora.yaml` 不存在，需 `cp configs/unichiflow_lora.example.yaml configs/unichiflow_lora.yaml`
- Backbone 基线已产生 `outputs/<ts>/checkpoints/best_model.pt`
- `data/d_pep_structures/` 已在本地（✅ 有）

**命令**：
```bash
python -m src.experiments.train_lora_real_d \
    train.max_iters=500 train.seed=2024 \
    sample.ckpt_path=outputs/<backbone_ts>/checkpoints/best_model.pt \
    2>&1 | tee baseline_logs/lora_step2.log
```

### 3. 本机（Apple M4 Pro）CPU smoke —— 可作为 Step 3 回归参照

**现状**：已跑通。环境、依赖、数据都就位，20 步 ~3 秒。

**环境**：conda env `unichiflow`（torch 2.11 + MPS 可用、但 smoke 用 CPU 更稳）
依赖：`pip install hydra-core omegaconf biopython lmdb joblib tqdm easydict scipy tensorboard`（`esm` 可跳过，backbone 默认 `esm.enable: false`）

**冒烟命令**（从项目根目录）：
```bash
/path/to/unichiflow-env/python -m src.experiments.train_unichiflow \
    train.device=cpu train.use_amp=false train.batch_size=2 \
    train.max_iters=20 train.save_freq=1000 train.num_workers=0 \
    train.logdir=/tmp/smoke_logs \
    data.use_subset=true data.subset_fraction=0.01 \
    data.cache_dir=.cache/pep_cache_subset_smoke \
    flow.sym_warmup_enable=false \
    hydra.run.dir=/tmp/smoke_logs/hydra
```

**首次运行会做**：
- `PepMergeSubsetDataset._load_structures` walk `data/pepmerge/`（~10348 候选），取 `subset_fraction=0.01`（~103 候选），joblib 并行过 `parse_pdb` + 构 LMDB（~2 秒）
- 真实样本过滤率 ~70%（PepMerge 里许多 peptide 链长 1–2，不满足 `>=3` 条件）
- 最后实际拿到 ~30 个有效训练样本，bs=2 下反复 shuffle

**预期行为**（seed=2024 下固定）：
- 17 次 ok step + 3 次 `skip_grad_nonfinite`（发生在 step 6/10/14 附近）
- Loss 从 16.56 起步，在 9.2~127 范围大幅抖动（bs=2 在 60–500+ 残基的极度异质 batch 上,训练初期正常现象）
- `grad_norm` 偶尔冲到 inf 触发 skip，这是 Y1 `_check_and_clip_gradients` 按设计接住

**用途**：
- Step 3 / Step 5 重构时的 bit-level 回归参照（refactor 前后 loss/grad/skip trace 逐行 diff 应完全相等）
- 新开发机验环境、快速 sanity（30 秒内确认 forward/backward/AMP-off 兜底/NaN-skip 全链路通）

**不能当**：真正的训练基线（样本量、步数都不够）、loss 下降趋势证据（20 步、bs=2、30 有效样本反复 shuffle，抖动噪声远大于信号）

### 4. 基线对比 —— 反馈什么回来

跑完把下面这些给我，Step 3 的抽象策略才好定：

1. 冒烟测试 exit code + 有无 traceback
2. 第 500 / 1000 步 tqdm 描述最后一行（`loss / cfm / sym / seq / grad_norm` 数值）
3. `non-finite` skip 事件次数（log 里 `grep "non-finite"`）
4. Backbone 总 wall-clock（用来之后比 Step 3 重构 + R6 OT 并行有没有真提速）
5. `outputs/<ts>/` TensorBoard 里 loss 曲线走势（下降 / 抖动 / 停滞）

### 5. Step 3 抽完后的回归验证

重跑一次**一模一样的命令 + seed**，对比：

| 指标 | 容忍度 |
|---|---|
| 前 10 步的 loss 逐值 | bit-level 或 ≤1e-5 相对误差 |
| 1000 步最终 loss | ≤5% 相对漂移 |
| grad_norm 量级 | 同阶即可 |
| skip 率 | 差值 ≤1 次 |
| checkpoints 大小 | 相等（同一 state_dict 结构） |

任何一条超标 → 回滚 Step 3 的 commit 对着 diff 找差异。

---

## 五、理论不变量清单（这些**没改动**）

- **PH-Flow 奇偶等变性**：`parity_positions(x) = -x`；`parity_rotations(R) = R · diag(-1,-1,1)`；`parity_torsions(θ) = -θ`；`v(P·x_t) = P·v(x_t)` 作为 sym loss 目标
- **Hard Geometric Decoder**：`Mirror(L_geom(θ)) = D_geom(-θ)` 的 Z-flip 恒等式
- **ChiralBridge LoRA gating**：chirality token 控制 LoRA 残差开合
- **Minibatch Riemannian OT**：Hungarian 求解，直线轨迹
- **SE(3) × T^k flow matching 目标**：x1-prediction + smooth-L1 on (trans, rot_log, ang_log) velocities

---

## 六、修改文件索引

| 文件 | 类型 | 涉及修复 |
|---|---|---|
| `src/models/chiralformer.py` | 核心 | R1, R3, R4, R7, R8, Y4, Y5, Y9, Y10 + P1 (OT 批量化) + M3–M6 (no_grad 诊断指标) |
| `src/flow.py` | 核心 | R5, R7 |
| `src/data/so3.py` | 核心 | R2 |
| `src/data/torus.py` | 核心 | R2, Y7 |
| `src/data/minibatch_ot.py` | 核心 | R6, Y8（threadpool 现在真正被 P1 调用到）|
| `src/models/semantic_conditioner.py` | 核心 | Y6 |
| `src/experiments/train_unichiflow.py` | 脚本 | R4, Y1, Y2 + Step 2 wiring + Step 3 do_train_step dispatch + M1–M6 TB 写入 |
| `src/experiments/train_lora_real_d.py` | 脚本 | R4, Y1 + Step 2 wiring（Step 3 未迁移，留给 Step 5）|
| `scripts/inference_unichiflow.py` | 脚本 | R4 |
| `src/experiments/components.py` | **新增** | Step 2 工厂 |
| `src/experiments/train_step.py` | **新增** | Step 3 `do_train_step` + `StepConfig` + `StepOutcome` |
| `configs/unichiflow_*.yaml`（8 个） | 配置 | C1（`max_len: 512 → 1024`） |

---

## 七、下一步决策点

Step 3 落地后，两条路：

**A. 做 Step 4（配置 schema 统一，中度风险）**
把 `use_amp / grad_accum_steps / sym_warmup_* / freeze_backbone / ckpt_path` 等字段在 backbone/4060ti/lora 三套 config 中显式对齐。这一步是 Step 5 Trainer 基类的前置（`BackboneTrainer` 和 `LoRATrainer` 共享 schema 才能继承）。

**B. 直接做 Step 5（Trainer 基类 + LoRA 对齐，高收益高风险）**
跳过 schema 统一，直接抽 `Trainer` 基类把 LoRA 接入 `do_train_step`（会给 LoRA 带来 AMP / grad_accum / LR decay 能力）。需要在同机器上先为 LoRA 建一个 bit-level baseline。

**当前建议**：A。Step 5 涉及 LoRA 行为变更，没有 GPU 基线数据就不好判定回归是否有漂移。Step 4 纯 yaml 改动 + 显式化 getter，风险近 0，能独立推进。

GPU 基线（§四.1 / §四.2）在 Step 5 之前必须跑，否则 LoRA 重构没有回归参照。CPU smoke 只能保 backbone。
