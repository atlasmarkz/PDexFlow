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

### 🟡 Y 类（9/10 完成，Y3 进行中）

| ID | 标题 | 核心改动 | 主要文件 |
|---|---|---|---|
| **Y1** | fail-loud NaN 处理 | 两个训练脚本 `_sanitize_gradients`（静默 `nan_to_num_`）→ `_check_and_clip_gradients`，检测到 NaN/Inf 直接 skip opt step + bad_step 计数 + LR decay | `train_unichiflow.py`, `train_lora_real_d.py` |
| **Y2** | early stopping 改名 + 澄清 | `EarlyStopping` → `BestCheckpointKeeper`（实际作用是保存最佳 ckpt，stop 分支几乎不触发），docstring/日志澄清 | `train_unichiflow.py` |
| **Y4** | 超长序列 assert | `predict_x1` 入口 `L > max_len + 8` 显式 raise，替换原本静默 clamp（会把不同残基映射到同一位置嵌入） | `chiralformer.py` |
| **Y5** | Transformer 空序列保护 | `_safe_padding_mask` 静态方法：某 batch row 全为 padding 时，把 pos 0 翻成 valid，避免 `nn.TransformerEncoder` 全 mask 产 NaN | `chiralformer.py` |
| **Y6** | ESM2 fp32 外壳 | `ESM2Conditioner.forward` 套 `torch.autocast(enabled=False)`，pretrained 权重在 fp32 下跑，避开 fp16 softmax 溢出 | `src/models/semantic_conditioner.py` |
| **Y7** | torus logmap 换 atan2 形式 | 已经是 atan2 实现（`tor_logmap = atan2(sin(Δ), cos(Δ))`），确认无需再改 | `src/data/torus.py` |
| **Y8** | OT fallback 修复 | scipy 缺失时的 `_greedy_permutation`：逐行贪婪，已选列置 `+inf`，保证输出是合法排列（原本 `argmin` 会出现重复列 → gather 错乱） | `src/data/minibatch_ot.py` |
| **Y9** | ChiLoRA gate buffer 时序修复 | `_set_chi_lora_gate` 加 assert + `_chi_lora_gate` 上下文管理器；预防 sym/main 支路串用 gate buffer | `chiralformer.py` |
| **Y10** | ChiLoRAConfig 解析显式化 | 显式接受 `Mapping / DictConfig / ChiLoRAConfig`；类型不对 `TypeError`，不再吞异常回落到 disabled | `chiralformer.py` |
| **Y3** | Trainer 抽象 | **进行中** — 见下文 Trainer Roadmap | — |

---

## 三、Trainer 抽象（Y3）五步 Roadmap

| 步骤 | 内容 | 状态 |
|---|---|---|
| **Step 1** | 固定基线（跑 backbone ~1000 步 + LoRA ~500 步，记录 loss/grad/skip 曲线） | ⏳ 待做（需 GPU 机器 + 数据） |
| **Step 2** | 抽纯函数工厂（dataset/dataloader/model/optimizer/ckpt-load） | ✅ **已完成** |
| **Step 3** | 把 train step 抽成纯函数 `do_train_step(batch, model, optim, scaler, cfg, ctx) -> StepOutcome`（AMP / grad_accum / NaN skip / LR decay 全塞进去） | ⏳ 待做 |
| **Step 4** | 统一 config schema（显式化 `use_amp / grad_accum_steps / sym_warmup_* / freeze_backbone / ckpt_path` 等） | ⏳ 待做 |
| **Step 5** | 抽 `Trainer` 基类，2–3 个 hook（`build_dataset / on_log / on_save`）；`BackboneTrainer` 和 `LoRATrainer` 作为子类 | ⏳ 待做 |

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
- `outputs/<timestamp>/` —— TensorBoard：`loss / smooth_loss / cfm_loss / sym_loss / seq_loss / grad_norm / lr`
- `outputs/<timestamp>/checkpoints/` —— `best_model.pt` + `500.pt` + `1000.pt`
- `baseline_logs/backbone_step2.log` —— tqdm 流（含 skip 事件、LR decay 事件）

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

### 3. 本机（Apple M4 Pro）能做的冒烟（可选）

**限制**：
- 本机无 torch/hydra/esm/scipy，需从零装环境（~500 MB，不含 fair-esm）
- `data/pepmerge/` 不在本地 → backbone 基线跑不了
- `outputs/checkpoints/*.pt` 不在本地 → LoRA 基线跑不了
- `amp.GradScaler(device="cuda", enabled=False)` 在无 CUDA 机器会抛错，需改代码
- CPU 模式 ~5–10 秒/step，1000 步 ≈ 1.5–3 小时

**结论**：本机只能做**合成数据冒烟**（随机 tensor 跑 5 步 forward+backward，断言 loss 有限），**不是基线**。性价比不如直接上 GPU。**当前决定：放弃本机冒烟。**

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
| `src/models/chiralformer.py` | 核心 | R1, R3, R4, R7, R8, Y4, Y5, Y9, Y10 |
| `src/flow.py` | 核心 | R5, R7 |
| `src/data/so3.py` | 核心 | R2 |
| `src/data/torus.py` | 核心 | R2, Y7 |
| `src/data/minibatch_ot.py` | 核心 | R6, Y8 |
| `src/models/semantic_conditioner.py` | 核心 | Y6 |
| `src/experiments/train_unichiflow.py` | 脚本 | R4, Y1, Y2 + Step 2 wiring |
| `src/experiments/train_lora_real_d.py` | 脚本 | R4, Y1 + Step 2 wiring |
| `scripts/inference_unichiflow.py` | 脚本 | R4 |
| `src/experiments/components.py` | **新增** | Step 2 工厂 |

---

## 七、下一步决策点

跑完 Backbone/LoRA 基线后，两条路：

**A. 继续做 Step 3（推荐）**
把 step 主循环抽成 `do_train_step`，预计能再去掉两个脚本里 ~40 行重复代码；风险中等，需严格回归对比。

**B. 停在 Step 2，直接上线**
当前状态已经修完所有 R/Y（除 Y3）；两个脚本的"setup 期"已统一，主循环维持原样。如果训练侧不需要再做大变更，可以就停在这里。Y3 后续可以单独开一轮。

选 A 的前提是 Step 1 基线必须先跑出来，否则没有回归参照。
