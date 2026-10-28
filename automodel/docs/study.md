# NeMo AutoModel 设计思路与执行流程调查报告

—— 以 GLM-5.3-Flash 9×H20 全参训练 与 DeepSeek-V4.1-Flash 8×H20 LoRA 两个生产案例为线索

> 调查范围：NeMo AutoModel 仓库工作副本，参考官方文档
> [DeepSeek-V4.1-Flash model-coverage](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/large-language-models/deepseek-ai/DeepSeek-V4.1-Flash)
> 与 [GLM-5.3-Flash model-coverage](https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/thudm/GLM-5.3-Flash)。

---

## 0. 摘要

NeMo AutoModel 是一个 **PyTorch 原生**（非自建算子栈）的大模型训练框架。它的核心设计是：**YAML 驱动 + `_target_` 反射实例化 + 类型化配置自建（config-owned construction）**，用一层薄薄的 HF Transformers 桥接（`NeMoAuto*` 包装类 + 架构注册表 + 能力声明）复用上游模型实现，把分布式（FSDP2/EP/CP/PP）、量化、PEFT、checkpoint 等基础设施以**固定的手术顺序**施加到模型上。

两个生产案例验证了同一框架的两端能力：

| | 案例 A：GLM-5.3-Flash | 案例 B：DeepSeek-V4.1-Flash |
|---|---|---|
| 模型 | 320B 总参/18B 激活的 **VLM**（KDA+DSA 混合注意力、mHC 四流、288 路由专家） | ~552B MoE **LLM**（CSA2 共享压缩 KV、384 路由专家、~196B Engram N-gram 表） |
| 微调方式 | 全参（冻结视觉塔） | LoRA r8/α32（基模全冻结） |
| 拓扑 | 9×H20=72 卡，**EP72 + CP2**，DP36，梯度累积 4 | 8×H20=64 卡，**EP32 + ep_shard2**，DP64，无累积 |
| 注意力后端 | cudnn（FlashMLA 前向 + cuDNN Frontend 反向） | tilelang（按 shape JIT 稀疏核） |
| 权重来源 | FP8 checkpoint → 加载期反量化 BF16 | FP8 dense/FP4 experts/FP8 Engram → 反量化 BF16 |
| 参照指标（官方 H100） | loss@step99≈1.23，峰值 57.7 GiB/卡，cudnn 比 sdpa 快 13.26% | train 2.2655 / val 2.5737，峰值 43.9 GiB/卡 |

两案例在内部集群上的实际运行均采用 `DISPATCHER=torch`（专家分发走 DTensor all-gather/reduce-scatter 回退路径，而非官方 recipe 的 HybridEP/DeepEP），这是与官方验证拓扑的唯一运行时差异，属于可移植性回退，机制上等价、通信效率略低。

---

## 1. 项目设计思路

### 1.1 定位：PyTorch-native，"组装"而非"重写"

框架不自建 Megatron 式的平行模型栈，而是站在 PyTorch 2.8+ 的原生分布式原语上：
FSDP2（`fully_shard` per-parameter sharding）、DTensor、`device_mesh`、torchrun。
模型本体尽量复用 HF `PreTrainedModel` 生态，自定义模型（如 `glm5_next`、`deepseek_v41`）也继承 `PreTrainedModel`，只是把权重布局和并行语义接管过来。

### 1.2 分层架构与单向依赖

```
automodel <config.yaml>                     # 或 torchrun -m nemo_automodel.cli.app
    │
    ▼
nemo_automodel/cli/app.py                   # 解析 recipe 目标，选择 launcher
    │
    ▼
recipes/ (llm/train_ft.py, vlm/finetune.py…)  # 训练入口：setup() 建组件 + run_train_validation_loop()
    │
    ▼
components/                                 # 自包含构建块：models/datasets/distributed/
    │                                         # checkpoint/quantization/_peft/moe/…
    ▼
_transformers/ (auto_model/registry/capabilities/infrastructure)
_diffusers/                                 # HF 桥接层
```

关键约束：**components 之间不互相隐藏 import**，只被 recipe 组合；recipe 保持"薄编排器"，实际构建交给各类型化 config 的 `build()`（AGENTS.md 的 Config-Owned Construction 原则）。

### 1.3 YAML + `_target_`：声明式配置的统一语法

所有 YAML 配置块用 `_target_` 指定 Python 可调用对象，兄弟键成为其 kwargs（Hydra/OmegaConf 风格），例如：

```yaml
optimizer:
  _target_: transformer_engine.pytorch.optimizers.fused_adam.FusedAdam
  lr: 1.0e-5
```

解析边界是 `components/config/loader.py:336` 的 `ConfigNode`（`instantiate()` 反射构造）和 `recipes/_typed_config.py` 的 `RecipeConfig`（把 `wandb:`/`tensorboard:`/`step_scheduler:` 等 YAML 段就地物化为类型化 config 属性）。规则是：**声明式状态留在 config 字段，运行时对象（进程组、mesh、tokenizer、device）作为 `build(...)` 的显式参数传入**，config 不得缓存运行时对象。

### 1.4 HF 桥接：注册表 + 能力声明 + 手术顺序

- `_transformers/registry.py` 的 `MODEL_ARCH_MAPPING` 按 HF `architectures` 字符串路由到本仓模型类；`_CUSTOM_CONFIG_REGISTRATIONS` 把 `glm5_next`、`deepseek_v41` 等本地 config 注册进 `AutoConfig`（免 `trust_remote_code`，且不依赖新版 transformers 包含该 model_type）。
- 模型类用静态嵌套 `ModelCapabilities` dataclass 声明并行能力（如 GLM/DSv41 均为 `tp=False, cp=True, pp=False, ep=True, thd=True`），`attach_capabilities_and_validate` 在构建时校验 YAML 拓扑与模型能力匹配（`_transformers/auto_model.py:644`）。
- **基础设施按固定顺序施加**（`_transformers/infrastructure.py:565` `apply_model_infrastructure`，与 skill 文档一致）：

```
PEFT(LoRA) → FP8 → QAT → checkpoint 加载 → 参数冻结/trainability → sharding(FSDP2/EP) → 设备放置 → torch.compile → CP 钩子
```

LoRA 在 shard **之前**注入（`infrastructure.py:664-671`），adapter 参数随基模一起进入 FSDP unit；这是保证 DCP 权限名（FQN）稳定的关键。

### 1.5 分布式设计：单一 MeshContext，EP 是"视图"不是"轴"

这是本框架最有辨识度的设计（`components/distributed/mesh.py`、`mesh_utils.py`）：

- 主 device mesh 形状 `(pp, dp_replicate, dp_shard, cp, tp)`，`dp = world/(tp·cp·pp)` 推导；
- **EP 不占主 mesh 轴**：把非 PP 轴 flatten 后 unflatten 成 `(ep_shard, ep)` 二维 **moe_mesh**（`mesh_utils.py:380-427`）。即 EP 是同一批物理 rank 对专家权重的另一种切分视角；
- `ep_shard_size = dp·cp·tp / ep_size` 自动推导。DSv41 H100/H20 案例中 WORLD=64、EP=32 → ep_shard=2：384 个专家先按 EP 维切（12 专家/rank），再在 ep_shard 维上 FSDP 分片（等效每 rank 6 个专家的参数量），把冻结基模常驻显存从 ~47GB 压到 ~23GB——**这是 552B 模型塞进 80GB 卡的核心机制**，用户无需配置，拓扑给出即推导；
- FSDP2 分片**没有 auto wrap policy**，而是按模块结构递归（`parallelizer.py:655-799`）：每个 decoder layer 一个 FSDP unit，逐层 `reshard_after_forward=True`（PP 时全 False），并手工设置 prefetch 链；MoE 专家单独 `fully_shard(moe_mesh[("ep_shard",)])`，block 级 FSDP 用 `ignored_params` 排除专家参数、按 fp32/bf16 dtype 分组（`moe/parallelizer.py:1005-1064`）；
- CP>1 时 FSDP 切到 `dp_shard_cp` 复合轴（**CP rank 即 DP shard rank，参数按 CP 维再切**），并启用 unused-param reduce-scatter；
- 梯度累积的正确性由 `MoEFSDPSyncMixin`（`moe/fsdp_mixin.py:151`）保证：训练循环在累积期间对所有 FSDP unit 关闭 `requires_gradient_sync`，最后一个 microbatch 统一开启（见 §2.8/§3.7）。

### 1.6 内核后端可插拔：BackendConfig

`BackendConfig`（`components/models/common/utils.py:412`）把 attn/linear/norm/experts/dispatcher/rope 的实现选择从模型代码中抽出，进 YAML：

```yaml
backend:
  attn: cudnn        # GLM: FlashMLA 前向 + cuDNN Frontend 反向
  attn: tilelang     # DSv41: 按 shape JIT 的稀疏注意力核
  dispatcher: hybridep | torch
  gate_precision: float32
```

个体层永不硬编码后端。`dispatcher=torch` 是不装 DeepEP 时的回退：`GroupedExperts` 用 DTensor all-gather/reduce-scatter 收发变长专家 token（`moe/experts.py:730`）；`hybridep` 走 `GroupedExpertsDeepEP` + nvshmem token dispatcher（`moe/megatron/token_dispatcher.py:1052`）。

### 1.7 权重转换与量化解码：state_dict_adapter

每个自定义模型配 `state_dict_adapter.py`，承担 HF↔native 的 key 重映射、张量布局转换和**加载期反量化**。两个案例的 checkpoint 都是量化发布（GLM: FP8 128×128 块；DSv41: FP8 dense 32×32 块 / FP4 experts / FP8 Engram），`dequantize_base_checkpoint: true` 使 adapter 在加载时解码为 BF16 训练。

### 1.8 Checkpoint 双轨制

`components/checkpoint/checkpointing.py`：训练态走 **DCP**（safetensors 格式时用 `_HuggingFaceStorageWriter` 写分片 safetensors），交付态走 **consolidated safetensors**（默认 `final`，合并分片导出 HF 权重 + 离线合并脚本 `consolidate.sh`）；PEFT 走第三条旁路——rank0 单卡写 `adapter_model.safetensors` + HF 格式 `adapter_config.json`。`LATEST`/`LOWEST_VAL` 指针异步发布。

### 1.9 lab 生产化层：模板 + 渲染 + torchrun

`lab/` 是离线集群的复现工作区，模式统一：**模板 YAML（官方 recipe 衍生）+ `RENDER_*` 环境变量渲染实体 YAML + 每节点 torchrun**。渲染器 `lab/scripts/render_config.py` 只依赖 PyYAML，`__SET_VIA_<VAR>__` 占位符未渲染时会在加载路径处立即失败（快速暴露而非静默下载）。启动器还负责：uv 自举、editable 安装、依赖钉版对齐、后端依赖检查（缺失即退出、不静默降级）、NCCL 日志分离（`NCCL_DEBUG_FILE` 每进程一个文件）、TensorBoard 兜底安装。

---

## 2. 案例 A：GLM-5.3-Flash 9×H20 全参训练

### 2.1 命令与脚本层

```
MODEL=…/GLM-5.3-Flash DATASET=…/MedPix-VQA DISPATCHER=torch bash lab/h20_glm53_flash_9node.sh
```

`lab/h20_glm53_flash_9node.sh` 每节点执行一次，流程：拓扑校验 → uv 自举 + editable 安装 → cudnn 后端依赖检查（flash_mla 可导入性）→ TensorBoard 检查 → 渲染模板 → torchrun。

**拓扑校验**（脚本 79-83 行，把失败前移到 shell 层）：
`WORLD=72` 须被 `EP=72` 整除；`EP=72` 须整除 288 个路由专家；`GBS=144` 须为 `DP=WORLD/CP=36` 的整数倍。

**渲染结果**（`lab/configs/glm5_3_flash_medpix_h20.yaml` + 环境变量）：
`MODEL/DATASET` 写入路径占位符，`DISPATCHER=torch` 覆盖 `model.backend.dispatcher`（默认 hybridep），`CKPT_CPU_OFFLOAD=true`（默认）写入 `checkpoint.cpu_offload`，`TB_LOGDIR` 写入 `tensorboard.log_dir`。

最终启动（159 行）：

```
torchrun --nproc_per_node=8 --nnodes=9 --node_rank=$NODE_RANK \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:13742 \
  -m nemo_automodel.cli.app <rendered.yaml>
```

### 2.2 CLI → recipe → in-process 执行

1. `cli/app.py:main`（`cli/app.py:95`）读 YAML，`recipe: FinetuneRecipeForVLM` 经 `resolve_recipe_name`（`cli/utils.py:49`，正则扫描 `recipes/` 下所有 `*Recipe*` 类）解析为 `nemo_automodel.recipes.vlm.finetune.FinetuneRecipeForVLM`；
2. 无 `skypilot:`/`nemo_run:` 段 → `InteractiveLauncher`；检测到 `LOCAL_RANK` + `TORCHELASTIC_RUN_ID`（torchrun worker）→ **recipe 在本进程内运行，不再重启 torchrun**（`interactive.py:113-118`）；
3. `recipe_cls(config)` → `RecipeConfig` 包装 → `setup()` → `run_train_validation_loop()`。

### 2.3 setup()：组件构建（`recipes/vlm/finetune.py:492`）

按序：

1. `initialize_distributed`（NCCL，120 分钟超时）→ `setup_logging` → `StatefulRNG(seed=1234, ranked=True)`（各 rank 种子不同）；
2. `create_distributed_setup_from_config` → `MeshContext.build` → 主 mesh `(pp=1, dp_replicate=1, dp_shard=36, cp=2, tp=1)` + moe mesh `(ep_shard=1, ep=72)`（EP=72=DP36×CP2，每 rank 4 个路由专家）；
3. rank0 构建 loggers：wandb（disable）、**TensorBoard**（`TensorboardConfig.build` 仅 rank0 返回 `SummaryWriter`，并写入完整 run config）；
4. `build_model(...)`（下节）→ `optimizer`（`torch.optim.AdamW`，lr 5e-6，betas [0.9,0.95]，wd 0.1）→ `FusedLinearCrossEntropy`（融合线性 CE，不物化 logits，对 320B 模型的 lm_head 是显存关键）；
5. dataloader：`make_medpix_dataset`（`components/datasets/vlm/datasets.py:189`）+ packed_sequence 配置（pretokenize、pack_size 2048、packing_ratio 0.9、NEAT 打包格式、balance_media_tokens）+ `StatefulDataLoader`（num_workers=1, drop_last）；
6. `step_scheduler`：`grad_acc = 144/(1×36) = 4` 个 microbatch/优化步；`ckpt_every=100`、`val_every=1000`（100 步训练实际不触发验证）。

### 2.4 模型构建与基础设施（`_build_model`，`auto_model.py:386`）

- `_target_: NeMoAutoModelForImageTextToText.from_pretrained` → registry 命中本地 `Glm5NextForConditionalGeneration`（`registry.py:101`），`is_custom_model=True`；
- world>1 → **meta 设备初始化**（`init_empty_weights`，不实际分配显存），权重留给基础设施阶段按 FSDP 语义加载；
- `attach_capabilities_and_validate`：校验 `cp=2, ep=72` 与模型声明（`supports_cp=True, supports_ep=True, supports_pp=False`）匹配；
- `apply_model_infrastructure`（`infrastructure.py:565`）本案例的实际序列：
  1. **无 PEFT**；2. **加载 checkpoint**：`Glm5NextStateDictAdapter.from_hf`（`state_dict_adapter.py:226`）——丢弃第 46 层 MTP 权重、`weight+weight_scale_inv` 成对 FP8 128×128 块反量化为 BF16、key 重映射（`hc_*`→mHC、KDA `A_log/dt_bias`→`_fp32_params`）、split→grouped 专家聚合、fp32 保留名单（`_fp32_params/e_score_correction_bias/rotary_pos_emb`）；
  3. **冻结**：`freeze_config` 冻结 vision tower（与 audio tower），语言模型全参可训练；
  4. **sharding**：`_shard_ep_fsdp` → MoE parallelizer：专家按 EP 维 Shard(0) DTensor 化 + `GroupedExperts`（torch dispatcher）注册，每层 block FSDP（bf16/fp32 分组），专家参数从 block FSDP 排除，根模型再包一层 FSDP（`wrap_outer_model: true`）；
  5. **AC**：whole-layer `checkpoint_wrapper`（NO_REENTRANT），`ignore_router_for_ac: true` 保存 router 输出避免重算时重路由；
  6. **CP 钩子**：模型 `_owns_cp_attention=True`，`prepare_model_inputs_for_cp` 安装 `shard_batch_for_glm5_next_cp`（连续切分 packed 序列）。

### 2.5 前向计算链（每 microbatch）

1. **数据**：NEAT 打包的 2K 序列（含图像 token 展开后的多模态序列）经 `shard_batch_for_glm5_next_cp` 按 CP 维连续对半切（`glm5_next/cp.py:258`），VLM 顶层先拼图像特征再切；
2. **processor**（离线 pretokenize 阶段）：`build_glm5_next_processor`（`processing.py:82`）组装本地 image processor + tokenizer，并修复首发 checkpoint chat template 的 image 占位（替换为 `<|begin_of_image|><|image|><|end_of_image|>`）；
3. **视觉塔**（frozen，无梯度）：Conv3d patch embed → 双轴 RoPE ViT → Conv2d 下采样 + PatchMerger，输出 splice 进 `image_token_id` 占位（`model.py:323`）；
4. **45 层解码器**（`layer_idx % 4 == 3` 为 DSA，其余 KDA，前 3 层 dense FFN）：
   - **mHC 四流超连接**：hidden 形状 `[B,S,4,H]`，Sinkhorn 迭代生成 pre/post/comb 混流权重，注意力/FFN 各一套投影（`layers.py:113`）；
   - **KDA 线性注意力**（FLA chunk 核 + 短卷积 + fp32 衰减门）；
   - **DSA 稀疏注意力**：KPool indexer 每 16 key 池化选 top-(2048/16)，`attn=cudnn` 时走 absorbed 形式（kv_b_proj 权重重排拆 w_kc/w_vc）→ `flash_mla_sparse_fwd` 前向 + cuDNN Frontend 反向（`common/cudnn_sparse_attention.py:180`）；**CP 下每层对局部 hidden 做可微 all-gather 恢复全序列 K/V**（`layers.py:633`）；
   - **MoE**：sigmoid_with_bias 路由（fp32 gate，noaux_tc），top-8 of 288 + 共享专家；`dispatcher=torch` → `GroupedExperts` DTensor all-gather/reduce-scatter；
5. **loss**：`FusedLinearCrossEntropy`，对 4 个流均值后的 norm 输出算 chunked CE。

### 2.6 训练循环（`train_ft.py` 同构的 vlm 版）

```
for batches in step_scheduler:            # 4 个 microbatch
    prepare_for_grad_accumulation(...)    # MoEFSDPSyncMixin: FSDP unit 关同步
    for i, batch in enumerate(batches):
        if i == last: prepare_for_final_backward(...)   # 开同步+reshard
        (local_loss * dp_size).backward() # DP 平均
    scale_grads_and_clip_grad_norm(1.0)   # EP 感知的全局范数
    optimizer.step(); zero_grad()
    update_moe_gate_bias()                # GLM 特有: noaux_tc 路由 bias 更新
    log → JSONL + TensorBoard(rank0)
    每 100 步 save_checkpoint
```

注意 `defer_fsdp_grad_sync: false`（与官方一致）：每个 microbatch 后即 reduce-scatter 并 reshard，**避免跨 4 个累积步保留全量梯度副本**——GLM 案例的显存关键取舍。

**Checkpoint**：DCP 分片 safetensors + `cpu_offload: true`（先搬 CPU 再写盘，规避本集群定制镜像的 torch 在 `tensor.to("cpu", non_blocking=True)` 处的 `CUDA error: invalid argument`——官方 recipe `enabled:false` 从未走过的路径，lab 实测发现并修复）；结束时 consolidated 导出 HF 权重。容量：全量模型单次保存 ≈1.9TB（BF16 模型 ~640GB + AdamW 态 ~1.27TB，按 17.5B mini 实测 96GB 外推）。

### 2.7 与官方 recipe 的差异（diff 验证）

| 项 | 官方（examples/vlm_finetune/glm5_next/…100steps.yaml） | lab 模板 |
|---|---|---|
| 模型/数据路径 | `zai-org/GLM-5.3-Flash`、`mmoukouba/MedPix-VQA`（Hub） | `__SET_VIA_*__` 本地路径渲染 |
| checkpoint | `enabled: false` | `enabled: true` + `cpu_offload: true` |
| ckpt_every_steps | 1000 | 100 |
| dispatcher | hybridep | 默认 hybridep，本次运行 `DISPATCHER=torch` |
| ci: 段 | 有（NVIDIA CI） | 移除 |
| tensorboard 段 | 无 | 新增（rank0 内建 logger） |
| chat_template | 显式 jinja 路径 | 由 processor 内建修复逻辑处理 |
| 拓扑/超参 | EP72/CP2/GBS144/100 步/cudnn | **完全一致** |

---

## 3. 案例 B：DeepSeek-V4.1-Flash 8×H20 LoRA

### 3.1 命令与脚本层

```
CKPT_SAVE_OPTIM=false TB_LOGDIR=<tb 日志目录> MODEL=…/DeepSeek-V4.1-Flash \
DATASET=…/hellaswag CKPT_DIR=<输出目录> DISPATCHER=torch bash lab/h20_dsv41_lora.sh
```

相对 GLM 启动器，DSv41 版增加三类生产防御：

- **离线开关**：`HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE=1`（杜绝 Hub 访问超时卡死）；
- **依赖钉版对齐**：当前代码要求 `transformers==5.18.0`（`resolve_revision`）与 `huggingface_hub==1.33.0`（`ResolvedRevision`），从内部镜像源对齐并做导入自检；`torchao` 是 `auto_model.py` 顶层硬依赖，只校验不自动装（PyPI 预编译 wheel 与镜像定制 torch 可能不兼容）；
- **物料前置校验**：`MODEL/config.json` 与 `DATASET/data/*.parquet` 在 torchrun 之前检查——路径错误在 shell 层暴露一次，而不是 64 个 rank 各报一次。

另有 DSv41 特有环境：`NCCL_PXN_DISABLE=1`（多节点 IB 上 PXN 退化路径的 all-to-all 字节偏移 bug，官方 H100 段落明确要求）、`TILELANG_CACHE_DIR` 指向节点本地 `/tmp`（tilelang 按 shape JIT 编译，缓存落在共享 home 会跨节点竞争，且大 cache 目录引发 NCCL 超时）。**每个 rank 首步触发 JIT 编译、之后读缓存，第一步显著偏慢属预期。**

用户实际覆盖项：`CKPT_SAVE_OPTIM=false`（只存 adapter 不存优化器态）、`TB_LOGDIR` 外部指定、`DISPATCHER=torch`。

### 3.2 拓扑与 YAML

`WORLD=64`、`EP=32`、CP/TP/PP=1 → 主 mesh dp=64；moe mesh `(ep_shard=2, ep=32)`。每 EP rank 12 个专家、跨 ep_shard 2 再分片 → 冻结 BF16 基模常驻 ~23GB/卡。`GBS=64=LBS×DP` → **梯度累积 1**（每步一个 microbatch）。

recipe 为 `TrainFinetuneRecipeForNextTokenPrediction`（`recipes/llm/train_ft.py:517`），执行链与 §2.2 相同。

### 3.3 模型构建：from_config 路径

与 GLM 的 `from_pretrained` 不同，DSv41 用嵌套 config 目标：

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_config
  config:
    _target_: …deepseek_v41.config.DeepseekV41Config.from_pretrained
    vision_config: {num_hidden_layers: 0}   # 声明式关闭视觉塔
```

`vision_config.num_hidden_layers: 0` 使 `DeepseekV41Model` 跳过 vision/aligner 构建（`model.py:305-311`）——同一模型类服务纯文本训练。`load_base_model: true` + meta 初始化后由基础设施阶段加载。

**量化解码**（`DeepseekV41StateDictAdapter`，`state_dict_adapter.py:243`）：dense 投影 FP8 E4M3 + 32×32 块 e8m0 scale；路由专家 FP4 E2M1（两个打包进 int8）；Engram 表 FP8。`from_hf` 丢弃 mtp.*/未建层/非本 rank 专家 → 逐张量 `dequantize_checkpoint_weight` 解码 BF16 → Engram 行 padding 对齐 owner 数。

### 3.4 LoRA 注入（本案例核心差异点）

`peft` 段 → `PeftConfig`（`components/_peft/lora.py:53`）：

- **target_modules 通配符**（`module_matcher.py:34` 把 `*`/`.*` 编译为正则，`model.layers.*.attn.wq_a` 不会误匹配 `.10`）：
  - `attn.wq_a/wq_b`（MLA 式低秩 q：hidden→q_lora_rank→heads）
  - `attn.wkv`（窗口 KV 投影）、`attn.wo_b`（输出投影）
  - `ffn.experts`（**MoE 专家整体**，经 `patch_moe_module` 换成 `GroupedExpertsLoRA`）
- `dim=8, alpha=32`（scale=4），`use_triton: true`（Triton 融合核；dtype 不匹配时自动降级 torch matmul）；
- **注入方式是 monkey-patch**：动态建 `type("PatchedLinearLoRA", (LinearLoRA, 原类))` 改 `__class__`（`lora.py:561`），权重 FQN 不变（DCP 兼容）；
- **注入时机在 FSDP shard 之前**（`infrastructure.py:669`），LoRA 参数随冻结基模进 FSDP unit；
- **trainability**：`param.requires_grad_("lora_" in name)`（`infrastructure.py:477`）——基模 552B 全冻结，可训练参数只有 lora_A/lora_B；
- 保存（见 §3.7）：rank0 单卡写 `adapter_model.safetensors`，绕过 DCP 全量收集。

### 3.5 前向计算链

1. **CSA2 注意力**（`attention.py:544`）：40 层分派 `compress_ratios=[0,0]+[2]*18+[1]*20+[0]*3`——Full 层（2/8/14/20）用 `_Compressor` softmax 门池化压缩 KV（nvfp4 量化存入跨层状态）；`_Indexer`（**冻结**，`requires_grad_(False)` + no_grad，硬 top-k 无语言模型梯度）对 index keys 打分选 top-512；Reuse 层直接复用状态的 compressed_kv 与 topk_indices；局部窗口 KV fp8。KV/index 的量化-反量化边界在每次前向保持，直通估计梯度（STE），无开关；
2. **tilelang 后端**：`_plan_sparse_indices` 排槽位（窗口槽在前、选中压缩位在后，64 对齐）→ vendored `sparse_attn_tilelang`（在线 softmax，可训练；`deepseek_v4/optimized_kernels.py:499`）；indexer 打分本身也有 tilelang 融合核（`indexer.py:30`，匹配参考实现的舍入边界）；
3. **Engram N-gram 表**（layers 1/14）：fast tokenizer 词表压缩 + 素数桶 XOR 哈希生成 2~4-gram 行 ID，查 owner-sharded embedding（~3.84 亿行 × 256 维 FP8，行 padding 到 owner 数=WORLD 64 的倍数）；表注册为 `Shard(0)` DTensor 并从 FSDP 忽略集排除（**Engram 是可训练的**，但 LoRA 场景下同样冻结）；
4. **mHC 单遍超连接**（Sinkhorn，fp32 系数与 BF16 残差共存——`mp_policy` 显式 `param_dtype: bfloat16, reduce_dtype: float32` 保持跨 FSDP 边界的 cast）；
5. **MoE**：sqrtsoftplus + noaux_tc 路由，top-6 of 384 + 共享专家；`DISPATCHER=torch` → `GroupedExperts` DTensor 路径；
6. **AC**：`DeepseekV41Block.nemo_checkpoint_context_fn`（`model.py:167`）在重算时**重放记录的 CSA2 选择与稀疏计划**（indexer 冻结但 top-k 选择必须与 forward 一致，否则 recompute mismatch）。

### 3.6 训练循环要点

- `prewarm: {cublas_backward: true, comm_groups: true}`：在分配器池尚小时预建 cuBLAS 工作区与通信子（`train_ft.py:820`），避免第 1 步显存峰值时才 JIT；
- `MaskedCrossEntropy`（逐 token 掩码 CE，num_label_tokens 全 DP all-reduce 归一）；
- `defer_fsdp_grad_sync: true` 但 GA=1——官方注释明确"one microbatch per update … memory is therefore not sensitive to defer_fsdp_grad_sync"；
- `moe.reshard_after_forward: true`（与 GLM 的 false 相反：DSv41 走 H100 段落建议，规避 torch 2.13 recompute mismatch 的组合之一）；
- 优化器 `FusedAdam`（TE）：fp32 master weights + bias correction，`exp_avg/exp_avg_sq` 均 bf16、`store_param_remainders: true`——LoRA 小参量下用低精度动量换带宽；
- lr：warmup 10 步 + cosine 衰减到 1e-6（min_lr），峰值 1e-5；
- 验证：每 50 步跑 HellaSwag validation 128 条（`val_every_steps: 50`），官方参照 val loss 2.5737。

### 3.7 Checkpoint：adapter 旁路

- `CKPT_SAVE_OPTIM=false` → `base_recipe.py:382` 跳过 `checkpointer.save_optimizer`，不写 `optim/`；
- PEFT 保存不走 DCP 全量路径：rank0 收集（`full_state_dict + cpu_offload + ignore_frozen_params`，PP/EP 模型走本地收集 + 跨 PP all-gather）→ 过滤只留 `lora_` 键 → 写 `<ckpt>/model/adapter_model.safetensors` + `adapter_config.json`（HF 最小格式，r/α/target_modules 从实际模块反推）+ tokenizer；
- 注释记录的坑：`save_optimizer=true` 时 DCP 对 optimizer state 混维度写条目在 `_validate_global_plan` 崩（IndexError）——这是 LoRA 复现场景推荐 false 的第二个理由（用户命令即如此）。

### 3.8 与官方 recipe 的差异（diff 验证）

| 项 | 官方（…lora_h100_8nodes.yaml） | lab 模板 + 本次运行 |
|---|---|---|
| 模型/数据 | `deepseek-ai/DeepSeek-V4.1-Flash`（Hub + revision pin） | 本地路径，revision 渲染为 **null**（本地快照的正确取值） |
| checkpoint | `enabled: false` | `enabled: true`；`save_optimizer` 默认 true，**本次 false** |
| 新增字段 | — | `cpu_offload`/`save_optimizer` 注释化开关 |
| ci: 段 | 有（NCCL_PXN_DISABLE 等） | 移除，改由启动器 export |
| dispatcher | hybridep | 本次 `DISPATCHER=torch` |
| 拓扑/超参 | EP32+ep_shard2/GBS64/100 步/LoRA r8 α32 | **完全一致** |

---

## 4. 执行流程 Side-by-Side 对照

### 4.1 相同的执行骨架

两案例共用同一条执行链，以下环节**逐阶段机制完全一致**：

1. **启动层**：lab 脚本每节点执行一次 → 拓扑校验（`NODE_RANK` 边界、EP 整除 WORLD 与路由专家数、GBS 为 DP 整数倍）→ uv 自举 + `--no-deps` editable 安装 → 后端依赖可导入性检查（缺失即退出、不静默降级）→ TensorBoard 兜底安装 → 模板 + `RENDER_*` 渲染实体 YAML（`__SET_VIA_*__` 残留检测）→ `torchrun --rdzv_backend=c10d -m nemo_automodel.cli.app`；
2. **CLI 层**：`cli/app.py:main` 读 YAML → `recipe:` 段经 `resolve_recipe_name` 解析 → `InteractiveLauncher` 检测 `LOCAL_RANK + TORCHELASTIC_RUN_ID` → **recipe 在 worker 进程内运行**（不重启 torchrun）；
3. **setup() 骨架**：`initialize_distributed`(nccl) → `StatefulRNG(seed=1234, ranked=True)` → mesh 构建（`MeshContext.build` → 主 mesh + moe_mesh）→ rank0 构建 TensorBoard writer → loss_fn → checkpointer → `build_model` → optimizer → StatefulDataLoader（dataset 构建包在 `FirstRankPerNode`）→ step_scheduler → lr_scheduler → restore；
4. **模型构建路径**：meta 设备初始化（world>1）→ registry 命中本地模型类（`MODEL_ARCH_MAPPING` + `_CUSTOM_CONFIG_REGISTRATIONS`）→ `attach_capabilities_and_validate` 校验拓扑 ↔ 能力声明 → `apply_model_infrastructure` 固定手术顺序（PEFT → 加载 → 冻结 → sharding → AC → CP 钩子）；
5. **权重加载**：量化发布 checkpoint → `state_dict_adapter.from_hf` 加载期反量化 BF16（fp32 保留名单：`_fp32_params`/`e_score_correction_bias` 等）→ mtp.* 丢弃；
6. **MoE 基础设施**：共用 `components/moe`（路由 fp32 gate、专家单独 FSDP wrap、`wrap_outer_model` 根模型再包、`MoEFSDPSyncMixin` 驱动梯度同步）；**本次两案例均 `DISPATCHER=torch`** → `GroupedExperts` DTensor all-gather/reduce-scatter 路径；
7. **训练循环结构**：`prepare_for_grad_accumulation` → 逐 microbatch `(loss × dp_size).backward()`（最后一个 microbatch 前 `prepare_for_final_backward`）→ `scale_grads_and_clip_grad_norm(1.0)`（EP 感知全局范数 + aux loss 缩放）→ `opt.step()/zero_grad()` → lr_scheduler.step → JSONL + TensorBoard(rank0) 记录 → 定期 ckpt / val / gc；
8. **收尾**：logger close → `_finalize_and_close_checkpointer` → SIGTERM 优雅退出路径。

### 4.2 逐阶段差异对照表

| 执行阶段 | 案例 A：GLM-5.3-Flash（9×H20 全参） | 案例 B：DeepSeek-V4.1-Flash（8×H20 LoRA） |
|---|---|---|
| **脚本层防御** | flash_mla 可导入检查（cudnn 后端） | 离线开关（`HF_HUB_OFFLINE` 等 3 项）、transformers/huggingface_hub 钉版对齐+导入自检、torchao 硬依赖校验、tilelang/cudnn/deep_ep 三分支检查、物料前置校验（config.json + parquet）、`NCCL_PXN_DISABLE=1`、`TILELANG_CACHE_DIR` 节点本地、`git pull` |
| **recipe 类** | `FinetuneRecipeForVLM`（`recipes/vlm/finetune.py:492`） | `TrainFinetuneRecipeForNextTokenPrediction`（`recipes/llm/train_ft.py:559`） |
| **模型入口** | `NeMoAutoModelForImageTextToText.from_pretrained`，路径直挂 `model.*` | `NeMoAutoModelForCausalLM.from_config` + 嵌套 `DeepseekV41Config.from_pretrained`；`vision_config.num_hidden_layers: 0` 声明式关视觉塔 |
| **拓扑** | WORLD=72，EP72（ep_shard=1，**4 专家/卡**），**CP2** → DP36 | WORLD=64，EP32 + **派生 ep_shard=2**（12 专家/rank 再对半，冻结基模 ~23GB/卡），CP1 → DP64 |
| **FSDP mesh** | FSDP 切 `dp_shard_cp` 复合轴（**CP rank 即 DP shard rank，参数按 CP 维再切**）+ unused-param reduce-scatter | FSDP 切 `dp` 轴；专家额外在 `moe_mesh[("ep_shard",)]` 上 Shard(1) 二次分片 |
| **量化解码** | FP8 dense 128×128 块（e4m3 + `weight_scale_inv`） | FP8 dense 32×32 块 + **FP4 experts**（int8 打包 nibble）+ **FP8 Engram 表** |
| **PEFT** | 无 | LoRA r8/α32 注入 5 类目标（`attn.wq_a/wq_b/wkv/wo_b` + `ffn.experts`），**shard 前 monkey-patch 改 `__class__`**，MoE 走 `GroupedExpertsLoRA` |
| **冻结策略** | `freeze_config` 冻结 vision/audio tower，语言模型**全参训练** | 基模 552B **全冻结**，trainability 靠 `"lora_" in name` 参数名约定 |
| **注意力后端** | cudnn：FlashMLA 前向 + cuDNN Frontend 反向（DSA 层）；KDA 层固定 FLA | tilelang：按 shape JIT 稀疏核（**每 rank 首步编译、显著偏慢属预期**），indexer 打分亦有 tilelang 融合核 |
| **注意力结构** | 45 层 **3:1 KDA:DSA** 混排（`layer_idx%4==3` 为 DSA），NoPE MLA absorbed 形式，KPool 每 16 key 池化选 top-128 | 40 层 **CSA2 三角色**（`compress_ratios=[0,0]+[2]*18+[1]*20+[0]*3`）：Full 层压缩 KV（跨层共享）、Reuse 层复用 top-512 索引、局部窗口 |
| **专有大模块** | mHC **四流**超连接（hidden `[B,S,4,H]`，Sinkhorn） | **Engram N-gram 表**（layers 1/14，~3.84 亿行 × 256 维 FP8，64 owner 行分片，可训练）；mHC 单遍实现 |
| **冻结子系统的梯度** | 视觉塔 no-grad；router 每步 `update_moe_gate_bias`（noaux_tc bias） | **indexer 整体 `requires_grad_(False)` + no_grad**（硬 top-k 无语言模型梯度）；AC 重算经 `nemo_checkpoint_context_fn` **重放 CSA2 选择**避免 recompute mismatch |
| **CP 数据路径** | `shard_batch_for_glm5_next_cp` 连续切分 packed 序列；DSA 层前向**可微 all-gather 恢复全序列 K/V** | 无 CP |
| **数据管线** | MedPix-VQA（VLM）：pretokenize + **NEAT 打包 2K**（packing_ratio 0.9、balance_media_tokens）+ processor（图像占位模板修复） | HellaSwag parquet（LLM）：`pad_seq_len_divisible: 64`、train 16384 条上限 + **val 128 条**（每 50 步验证） |
| **损失函数** | `FusedLinearCrossEntropy`（融合线性 CE，**不物化 logits**） | `MaskedCrossEntropy`（逐 token 掩码，num_label_tokens 全 DP 归一） |
| **优化器** | `torch.optim.AdamW`，lr 5e-6，**constant** | TE `FusedAdam`，lr 1e-5，**warmup 10 + cosine → 1e-6**；fp32 master weights + **bf16 动量** + remainders |
| **梯度累积** | **GA=4**（GBS144/DP36）；`defer_fsdp_grad_sync: false`——每 microbatch 即 reduce-scatter+reshard，避免跨 4 步驻留全量梯度 | **GA=1**（GBS64=DP64）；`defer_fsdp_grad_sync: true`（官方注明此时不敏感）；`moe.reshard_after_forward: true`（A 为 false） |
| **mp_policy** | 默认（无显式块） | 显式 `param_dtype: bf16, reduce_dtype: fp32`（单遍 mHC 的 fp32 系数跨 FSDP 边界保 cast） |
| **prewarm** | 无 | `cublas_backward + comm_groups`（分配器池尚小时预建，避第 1 步峰值 JIT） |
| **验证** | 无（`val_every: 1000` > max_steps 100） | 每 50 步 HellaSwag validation（`_run_validation_epoch`） |
| **checkpoint 保存** | **DCP 全量**：分片 safetensors（单次 ≈1.9TB，模型 ~640GB + AdamW 态 ~1.27TB）+ `cpu_offload: true`（规避定制镜像 torch 的 `CUDA error: invalid argument`）+ 结束 consolidated 导出 HF 权重 | **adapter 旁路**：rank0 单卡写 `model/adapter_model.safetensors` + HF 格式 `adapter_config.json` + tokenizer；本次 `CKPT_SAVE_OPTIM=false` 跳过优化器态（MB 级产物） |
| **官方参照（H100）** | loss@step99≈1.23，峰值 57.7 GiB/卡，cudnn 9,489 TPS（sdpa 8,378） | train 2.2655 / val 2.5737，峰值 43.9 GiB/卡 |

### 4.3 差异的根源解读

三组根因决定了上表的大部分差异：

1. **全参 vs LoRA** → 冻结策略、优化器选型（全量 AdamW 态 1.27TB vs TE 低精度动量）、checkpoint 路径（DCP 全量+consolidated vs rank0 adapter 旁路 + `save_optimizer=false`）、GA 敏感性（GLM 的 `defer_fsdp_grad_sync=false` 是 4 步累积下的显存关键取舍，DSv41 GA=1 无此问题）；
2. **显存拓扑的不同解法** → GLM 320B 用 **CP2 换 DP**（序列维对半切、参数按 CP 维再分、DSA 层可微 all-gather）；DSv41 552B+196B Engram 用 **ep_shard 再分片**（冻结专家跨 2 rank 对半，Engram 表按行 owner 分片到 64 rank）。两者都由框架从 `ep_size/cp_size` 自动推导，用户不接触 mesh 内部；
3. **VLM vs LLM / 模型结构** → recipe 类与数据管线（processor + 打包 vs tokenizer + padding）、注意力后端选择（GLM 的 DSA 用 cudnn/FlashMLA、KDA 固定 FLA；DSv41 全稀疏核走 tilelang JIT）、各自的专有模块（mHC 四流 vs Engram + 跨层 CSA2 状态）。

## 5. 对比观察与结论

### 5.1 生产化层的通用模式（lab/ 的工程价值）

两套启动器共享同一套方法论，且都被两个案例验证：

1. **失败前移**：拓扑整除性、物料存在性、后端依赖可导入性全部在 torchrun 之前的 shell 层校验；
2. **不静默降级**：tensorboard/tilelang/cudnn/deep_ep/torchao 缺失即退出并给出可退回的替代路径（如 `BACKEND=sdpa`/`DISPATCHER=torch`），由人决定降级；
3. **模板即文档**：模板 YAML 头注释逐条记录与官方 recipe 的差异及原因（为何 revision 渲染 null、为何 cpu_offload、为何 save_optimizer=false），占位符 `__SET_VIA_*__` 未渲染即快速失败；
4. **可观测性**：主日志（set -x trace）与 NCCL 日志（每进程一文件）分离，TensorBoard 由 recipe 内建 logger 在 rank0 进程内直写。

### 5.2 运行时差异提示

两次实际运行均使用 `DISPATCHER=torch`（官方验证拓扑为 hybridep/DeepEP）。机制上等价（`GroupedExperts` 的 DTensor all-gather/reduce-scatter 路径），正确性已由跑通验证，但专家 token 分发通信效率低于 DeepEP 的 nvshmem 内核——若后续追求官方参照吞吐（GLM cudnn 9,489 TPS / DSv41 43.9GiB 峰值同水平），且镜像内 deep_ep 可用（`lab/Dockerfile.h20` 已固化），可尝试切回 `DISPATCHER=hybridep`。

### 5.3 参照指标（官方 H100 实测，供 H20 结果对照）

| | GLM-5.3-Flash（EP72/CP2, 100 步） | DSv41-Flash LoRA（EP32, 100 步） |
|---|---|---|
| loss | step99 ≈ 1.23（cudnn）/1.2359（sdpa） | train 2.2655 / val 2.5737 |
| 峰值显存 | 57.7 GiB/卡 | 43.9 GiB/卡 |
| 吞吐 | cudnn 9,489 TPS vs sdpa 8,378 TPS（+13.26%） | —（H20 96G 余量更大） |

H20 与 H100 同为 SM90，内核（FlashMLA sm_90a、tilelang）可直接复用。

---

## 附录：关键文件索引

| 环节 | 文件 |
|---|---|
| CLI/recipe 解析 | `nemo_automodel/cli/app.py`、`cli/utils.py`、`components/launcher/interactive.py` |
| LLM/VLM recipe | `recipes/llm/train_ft.py`（setup:559, loop:1118, optim step:1465）、`recipes/vlm/finetune.py`（setup:492） |
| 配置边界 | `components/config/loader.py`（ConfigNode:336）、`recipes/_typed_config.py`（RecipeConfig:139） |
| 模型桥接 | `_transformers/auto_model.py`（_build_model:386）、`registry.py`、`infrastructure.py`（apply_model_infrastructure:565） |
| Mesh/并行 | `components/distributed/mesh.py`、`mesh_utils.py`（moe mesh:380）、`parallelizer.py`（FSDP 递归:655）、`moe/parallelizer.py`（ep_shard:1018）、`moe/fsdp_mixin.py`（梯度同步:151） |
| MoE | `components/moe/layers.py`（dispatcher 选择:813）、`experts.py`（GroupedExperts:730 / DeepEP:1454）、`megatron/token_dispatcher.py` |
| GLM 模型 | `components/models/glm5_next/{model,layers,vision,cp,state_dict_adapter,processing}.py`、`common/cudnn_sparse_attention.py` |
| DSv41 模型 | `components/models/deepseek_v41/{model,attention,engram,indexer,state_dict_adapter,layers}.py` |
| PEFT/量化 | `components/_peft/{lora,module_matcher}.py`、`components/quantization/` |
| Checkpoint | `components/checkpoint/{config,checkpointing,stateful_wrappers}.py` |
| 训练组件 | `components/training/{step_scheduler,rng,timers,utils}.py`、`components/loggers/loggers.py`（TensorboardConfig:490） |
| 数据 | `components/datasets/vlm/datasets.py`（make_medpix_dataset:189）、`llm/hellaswag.py` |
| lab 层 | `lab/h20_glm53_flash_9node.sh`、`lab/h20_dsv41_lora.sh`、`lab/scripts/render_config.py`、`lab/configs/*.yaml`、`lab/README.md` |
| 官方参照 | `examples/vlm_finetune/glm5_next/glm5_3_flash_medpix_packed2k_ep72_cp2_100steps.yaml`、`examples/llm_finetune/deepseek_v41/deepseek_v41_flash_hellaswag_lora_h100_8nodes.yaml` |
