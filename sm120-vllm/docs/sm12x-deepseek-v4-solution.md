# DeepSeek V4 在 SM12x 上的 vLLM 推理方案

> 作者：jasl（DeepSeek V4 SM12x 后端移植与优化）
> 涉及仓库：`vllm-project/vllm`（分支 `ds4-sm120-preview-dev`）+ 配套 `DeepGEMM-jasl`（分支 `sm120`）
> 整理日期：2026-06-16

本文档梳理 jasl 为 **DeepSeek V4 在 NVIDIA Blackwell SM12x GPU 上的 vLLM 推理支持** 所做的完整工程方案：根因、总体架构、两个仓库的分工、一次前向的数据流、DeepGEMM 的角色与替代品、分支模型、使用方式与完成度评估。

---

## 1. 一句话概述

把 DeepSeek V4 的推理支持从**数据中心 Blackwell（SM90/SM100，B200）**下沉到**消费/工作站/边缘 Blackwell（SM12x）**。这是一套**跨两个仓库、分两层**的方案：先给 DeepGEMM 补齐 SM12x 的张量核心原语（解除"DSv4 在 SM12x 起不来"的硬卡点），再在 vLLM 用可移植 Triton + 回退实现 + 手调配置把完整前向与服务化跑通。

## 2. 目标硬件

| 架构代号 | 设备 | 用途 |
|---|---|---|
| **SM120**（GB202） | RTX PRO 6000 Blackwell Workstation Edition | 工作站 |
| **SM121**（Grace-Blackwell） | GB10（DGX Spark / Jetson Thor） | 边缘/嵌入式 |

关键基础工作：把 CUDA major 12（SM120 + SM121）统一归一到 **`sm_120f` JIT target + `sm120` include 后缀**，使这两类设备共享同一套 SM12x 内核族。

## 3. 根因：为什么 SM12x 开箱跑不了 DSv4

DeepSeek V4 的前向路径依赖两类上游库，而**它们都只支持 SM90/SM100**：

1. **DeepGEMM** —— 提供 FP8/FP4 GEMM、fused MoE（Mega MoE）、paged MQA logits（闪电 indexer 打分）、HyperConnection(HC) prenorm GEMM 等张量核心原语。上游 README 明确写 *"NVIDIA SM90 or SM100 architecture GPU"*，**SM12x 设备会在 dispatch 时被直接拒绝**。
2. **FlashInfer / CUTLASS sparse MLA** —— 提供完整 sparse MLA 注意力，在 SM12x 上不可用或稳定。

DeepGEMM-jasl 的提交 `2206a1d` 把根因讲得很直白：

> *"DeepGEMM currently rejects SM12x devices in the DeepSeek V4 forward path because the HC prenorm GEMM and paged MQA logits families only dispatch SM90/SM100 implementations."*

因此 SM12x（RTX PRO 6000 Blackwell、GB10/DGX Spark）上 DSv4 根本无法启动，更谈不上服务化。

## 4. 总体架构：两层 + 两仓库

```
┌─────────────────────────────────────────────────────────────────┐
│ Repo 2: vLLM  ds4-sm120-preview-dev   （服务 / Triton 层）        │
│   • 可移植 Triton sparse MLA 全注意力（DeepGEMM 不做的部分）       │
│   • DeepGEMM-only 接口的 Triton/torch 回退（无 fork 也能跑）       │
│   • 手调 FP8 配置（SM12x 专用）                                   │
│   • kernel warmup / 调度器公平性 / MTP / reasoning parser / FP4   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ 调用 GEMM / MQA-logits / HC-prenorm 接口
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Repo 1: DeepGEMM-jasl  sm120   （CUDA 张量核心原语层）            │
│   • sm120 FP8 einsum / paged MQA logits / tf32 HC prenorm GEMM   │
│   • SM120 MMA 原语（避开 SM90 WGMMA / SM100 tcgen05+TMA 假设）     │
│   • SM12x JIT 派发启发式、sm_120f 统一 target                      │
└─────────────────────────────────────────────────────────────────┘
```

**设计哲学**：能复用上游算子的，用 DeepGEMM-jasl 给 SM12x 补真 CUDA 内核；DeepGEMM 没有的（完整 sparse MLA 注意力），在 vLLM 用可移植 Triton 重写；并为所有 DeepGEMM-only 接口都准备一份 Triton/torch 回退，使方案对"是否安装 DeepGEMM-jasl fork"具备鲁棒性。

---

## 5. Repo 1：DeepGEMM-jasl `sm120` 分支（CUDA 原语层）

- **时间**：2026-04-24 ~ 04-26（**先于 vLLM 侧**，因为这是"能否起跑"的硬卡点）
- **规模**：14 个提交（13 jasl + 1 ergodic-flow），约 **+2,035 行**
- **定位**：扩展 DeepGEMM 本体，为 SM12x 补齐原本只有 SM90/SM100 实现的算子族

### 5.1 核心改动

| 文件 | 作用 |
|---|---|
| `deep_gemm/include/deep_gemm/mma/sm120.cuh`（62 行） | SM120 矩阵乘累加原语，避开 SM90 WGMMA / SM100 tcgen05+TMA 的硬件假设 |
| `deep_gemm/.../impls/sm120_fp8_einsum.cuh`（126 行） | DSv4 O-proj 等 FP8 einsum |
| `deep_gemm/.../impls/sm120_fp8_paged_mqa_logits.cuh`（395 行） | sparse MLA indexer 的 FP8 paged MQA 打分（直接 dequant → FP32 累加 → ReLU → per-head weight，无效位置写 -inf） |
| `deep_gemm/.../impls/sm120_tf32_hc_prenorm_gemm.cuh`（288 行） | HyperConnection prenorm GEMM（保留 split-output 带 square-sum 的 ABI） |
| `csrc/jit_kernels/heuristics/sm120.hpp`（243 行） | SM12x JIT 内核派发启发式 |

### 5.2 基础设施改动

- **SM120/SM121 归一**：CUDA major 12 → `sm_120f` JIT target + `sm120` include 后缀（统一内核族）。
- **CUTLASS 升级**到 4.4.2。
- **参考回退（reference fallbacks）**：早期用 ATen CUDA matmul 提供 HC prenorm 等的正确性回退，后续替换为真内核。
- **回归测试**：`tests/test_sm120_kernels.py`（380 行）覆盖 HC prenorm GEMM 与 FP8 paged MQA logits（含 fused paged KV-cache 布局）。

### 5.3 仍 gated（未移植到 SM12x）

FP4 paged MQA、non-paged MQA、MegaMoE、通用 FP8/FP4 GEMM —— 这些在 SM12x 仍走 vLLM 侧的其他路径（CUTLASS scaled_mm / Marlin / Triton / FlashInfer）。

---

## 6. Repo 2：vLLM `ds4-sm120-preview-dev`（服务 / Triton 层）

- **时间**：2026-05-05 ~ 06-16（持续开发）
- **规模**：101 个 jasl 提交（另 2 个外部协作者各 1 个），约 **+14,500 / -381 行**，跨 106 个文件
- **定位**：把 DeepGEMM 补的原语 + 自研 Triton 注意力串成完整前向，并完成服务化加固

### 6.1 工作线一览

| 工作线 | 关键文件 | 内容 |
|---|---|---|
| **可移植 Triton sparse MLA**（核心） | `vllm/v1/attention/backends/mla/sparse_mla_kernels.py`（3521 行）、`sparse_mla_env.py` | 重写 sparse MLA 全流程：decode + prefill + D512 split/chunked prefill；SM12x 自动启用 |
| **DeepGEMM 接口回退** | `vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py`（706 行） | 为 mqa logits / paged mqa logits / tf32 hc prenorm 提供 torch/Triton 回退（无 fork 也能跑） |
| **MQA Triton 内核** | `vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py`（726 行） | paged FP8 MQA logits、不物化全 logits 的分块 row top-k |
| **手调 FP8 配置** | `vllm/model_executor/layers/quantization/utils/configs/*.json` 等 ~25 个 | 为 RTX PRO 6000 / GB10 调 dense GEMM 与 fused-MoE 的 FP8 W8A8 block tile |
| **FP4 MoE** | `mxfp4.py`、`nvfp4.py`、`routed_experts.py`、`flashinfer_cutlass_moe.py` | NVFP4 ModelOpt 路由、FlashInfer CUTLASS MXFP4 opt-in、MoE metadata/EPLB |
| **kernel warmup 基建** | `vllm/model_executor/warmup/kernel_warmup.py`（424 行） | 启动时预热 JIT Triton/DeepGEMM 内核，避免热路径 JIT |
| **调度器长 prefill 公平性** | `vllm/v1/core/sched/scheduler.py`（+153 行）、测试 +578 行 | 超长 prefill 不饿死在跑的 decode/prefill/缓存长 prompt 尾部；prefix-cache 块哈希修复 |
| **MTP spec decode** | `nvidia/mtp.py`、`vllm/v1/spec_decode/llm_base_proposer.py` | MTP 调度稳定化、workspace warmup、sparse SWA 重排、小批 cudagraph 挂起修复 |
| **服务化加固** | `vllm/reasoning/deepseek_v4_reasoning_parser.py`（304 行） | 长上下文（~95k–100k）下 DSv4 偶尔漏 `</think>` 的防御式补全、DSML tool-call 流式标记处理 |
| **加载与配置** | `weight_utils.py`、`config/vllm.py`、`config/compilation.py` | GB10 加载开销优化、权重过滤快速 safetensors、SM121 跳过 cudagraph 自动启用 |
| **持续 rebase 跟随上游** | 多处 | 上游 DSv4 在重构（MegaMoE、runner refactor、metadata split、#45061），大量 "after rebase" 清理 |

### 6.2 sparse MLA 的派发逻辑

`vllm/models/deepseek_v4/nvidia/flashmla.py` 中，`is_triton_sparse_mla_enabled(q.device)`（`flashmla.py:803/947`）控制是否走 Triton 路径。SM12x 上自动为 True（`sparse_mla_env.py` 的 `_is_sm12x_device` 检测 capability major==12）。

---

## 7. 一次 DSv4 前向在 SM12x 上的数据流

```
DSv4 forward on SM12x
│
├─ Dense FP8 GEMM（QKV / O-proj / dense 层）
│    └─ DeepGEMM SM120 fp8_einsum/gemm  ──(fallback)──► vLLM CUTLASS scaled_mm(SM120) / Marlin + 手调配置
│
├─ MoE 专家
│    └─ fused-MoE（手调 FP8 W8A8 block 配置）/ FP4: MXFP4(FlashInfer CUTLASS) | NVFP4(ModelOpt)
│       （DeepGEMM 的 MegaMoE FP8/FP4 在 SM12x 仍 gated → 走 vLLM 路径）
│
├─ Sparse MLA 注意力（核心）
│   ├─ compressor + indexer 打分
│   │   └─ paged MQA logits ─► DeepGEMM SM120 fp8_paged_mqa_logits
│   │                            └─(fallback)─► vLLM sm12x_mqa.py / sm12x_deep_gemm_fallbacks（Triton/torch）
│   ├─ HC prenorm GEMM ─► DeepGEMM SM120 tf32_hc_prenorm_gemm
│   │                       └─(fallback)─► vLLM _tf32_hc_prenorm_gemm_torch（ATen matmul + square-sum）
│   ├─ row top-k（不物化全 logits）─► vLLM sm12x_mqa.py Triton 分块 top-k
│   └─ 稀疏注意力（gather KV / softmax / merge）─► vLLM 可移植 Triton sparse_mla_kernels.py
│                                                  （DeepGEMM 无对应物，必须 Triton）
│
├─ MTP spec decode ─► vLLM proposer + warmup
│
└─ 全程：启动 kernel warmup 预热 JIT；运行期 调度器公平性 + prefix cache
```

---

## 8. DeepGEMM 的角色与替代方案

### 8.1 DeepGEMM 不是唯一选择

vLLM 里**每个原语都有替代后端**，且 SM12x 已有**原生非-DeepGEMM FP8 路径**（如 `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu`）：

| DeepGEMM 功能 | 替代后端 |
|---|---|
| FP8 GEMM（dense） | `scaled_mm/cutlass.py`（含 SM120 原生）、`scaled_mm/marlin.py`、`scaled_mm/pytorch.py`、`scaled_mm/flashinfer.py` |
| FP8/FP4 MoE | `triton_moe.py`、`triton_cutlass_moe.py`、`marlin_moe.py`、`flashinfer_cutlass_moe.py`、`mxfp8_native_moe.py`、`nvfp4_emulation_moe.py`、`trtllm_fp8_moe.py`、`trtllm_nvfp4_moe.py` 等十几个 |
| FP8 einsum | CUTLASS scaled_mm / Marlin / Triton |
| paged MQA logits | jasl 的 `sm12x_deep_gemm_fallbacks.py`（torch）/ `sm12x_mqa.py`（Triton） |
| HC prenorm GEMM | `_tf32_hc_prenorm_gemm_torch` |

**结论**：去掉 DeepGEMM-jasl，DSv4 在 SM12x 上照样能跑（dense 走 CUTLASS/Marlin SM120 FP8，MoE 走 Triton/FlashInfer，MQA logits 与 HC prenorm 走 torch/Triton 回退，sparse MLA 走可移植 Triton）。`sm12x_deep_gemm_fallbacks.py` 的存在正是为了让方案对 fork 有无具备鲁棒性。

### 8.2 为什么还要做 DeepGEMM fork

1. **性能**：DeepGEMM 是 DeepSeek 针对这些**精确 op 和 shape** 手调的张量核心内核，TFLOPS 高。替代品更慢，或**表达不了 DSv4 特有的融合算子**（paged MQA logits 的特殊 KV 布局 + per-head weight + 无效位 -inf；HC prenorm 的 split-output square-sum）。用替代品等于拆成"GEMM + 多个 reduction" → 多次访存、需物化中间 logits（回退路径用 `_SM120_MQA_LOGITS_MAX_SCORE_BYTES = 64 MiB` 钳制显存，正是这个开销的体现）。
2. **解除 stock DeepGEMM 拒绝 SM12x 的硬卡点**：DSv4 前向硬依赖 DeepGEMM 接口，而 stock DeepGEMM 在 SM12x dispatch 失败。fork 把"快路径"从"仅 SM90/SM100"扩展到 SM12x。

### 8.3 选型机制

vLLM 的 `scaled_mm` 与 `fused_moe` 是**多后端 + 按 arch/quant 自动选**。jasl 的工作不是强制绑定 DeepGEMM，而是**为 "SM12x 这个 arch" 补上各后端的可用项 + 手调配置**。

---

## 9. 分支模型与时间线

### 9.1 分支关系

| 分支 | HEAD | 时间点 | 上游 main 基点 | 性质 |
|---|---|---|---|---|
| `ds4-sm120-preview` | `e4923b2ea` Revert "Stabilize SM12x piecewise CUDA graphs" | 冻结 2026-05-10 | 落后 main **1331 个提交** | 早期冻结预览快照 |
| `ds4-sm120-preview-dev` | `5be22eb0e`（== 本地 `main`） | 活跃到 2026-06-16 | 落后 main **0 个提交**（已并入 main） | 持续开发主干 |
| `ds4-sm120-full` | — | — | — | （存在但未在本分析展开） |

**关键事实**：`ds4-sm120-preview-dev` 已完全 rebase 到最新 main 并并入本地 `main`。`preview` 的 20 个奠基提交基本都在 `dev` 里以不同 SHA 体现（rebase 重写了历史）。

### 9.2 时间线

```
04-24 ~ 04-26  DeepGEMM-jasl sm120 分支（补 SM12x 原语，解除起跑卡点）
05-05 ~ 05-07  vLLM 侧奠基：portable sparse MLA Triton + SM12x fallback ops + 串接前向
05-08 ~ 05-20  内核调优、warmup、MQA top-k、FP8 配置
05-21 ~ 06-16  调度器公平性、FP4 MoE、稳定化、rebase 跟随、清理（持续）
```

### 9.3 preview 独有但 dev 中未见的项（需确认）

- **`Support DeepSeek V4 pipeline parallel preview`**（Isotr0py，5/3）——在 dev 代码中按 `pipeline_parallel` 关键词无命中。dev 里 `deepseek_v4` 整个被重构（旧单文件 → 新包），PP 可能被吸收或暂未迁移。**如需 PP 建议单独确认。**
- `Stabilize SM12x piecewise CUDA graphs` + 其 Revert（在 preview tip 相互抵消，净效果未启用）——dev 走了另一条路（SM121 跳过 cudagraph 自动启用等）。

---

## 10. 使用方式

### 10.1 核心设计：零配置自动启用

在 SM12x（capability major == 12）上，sparse MLA 的 Triton 路径与相关开关**默认自动启用**，通常无需手动设置。DeepGEMM-jasl 需作为外部包安装（vLLM 优先用 site-packages 的 deep_gemm，见 `vllm/utils/deep_gemm.py` 的 `_import_deep_gemm`）。

### 10.2 主要环境变量（`vllm/envs.py`）

```bash
# ── sparse MLA 主开关 ───────────────────────────────────────
# 留空=SM12x 自动启用；0=强制关；1=强制开
VLLM_TRITON_MLA_SPARSE=0|1

# sparse MLA 调优旋钮（均有默认值，一般不用动）
VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE=512
VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE=256
VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=1|2|4
VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0|1

# ── D512 prefill 分块路径 ───────────────────────────────────
VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL=1
VLLM_DEEPSEEK_V4_INDEXED_D512_CHUNKED_PREFILL=1

# ── 启动预热（避免热路径 JIT）──────────────────────────────
VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP=1
VLLM_DEEP_GEMM_WARMUP=skip|full|relax
```

配合 vLLM 通用 flag：`--attention-backend`、`--attention-config.*` 等。

### 10.3 文档现状

- **无面向用户的 SM12x 专门文档/README/example**。jasl 在本分支从未碰 `docs/`。
- 最接近"使用说明"的信息源：`vllm/envs.py` 里各变量的注释、`sparse_mla_env.py` 的 docstring、提交信息。
- `docs/design/attention_backends.md` 的 "DeepSeek V4 Decode Backends" 小节只覆盖数据中心后端，Compute Cap. 写的是 "9.x-10.x"，**未涵盖 SM12x**。

---

## 11. 完成度评估

**整体处于"功能基本打通、进入收尾/加固/跟随上游"的后期阶段**（与分支名 `preview-dev` 一致）。

**依据**：
1. 奠基早（5 月初落库核心内核），5 月中旬后几乎全是 Tune / Stabilize / Fix / Warmup / 配置 / 清理。
2. 提交语义从 "Add/Wire" 转向 "Tune/Fix/Clean/Guard/Integrate"——典型成熟期 + 集成保持特征。
3. 仍有活跃的长尾修复（6/9–6/16）：prefix-cache 块哈希、MoE metadata sync、tool-call 标记、stray think-end —— **长上下文 + 工具调用 + prefix cache 组合下尚未完全稳定**。

**粗略量化**：核心功能约 **85–90% 完成**；剩余主要是长上下文边角稳定性、与上游持续 rebase 对齐、以及正式 upstreamable 化（独立成 PR、补文档/基准）。**pipeline parallel 是否完整支持待确认**。

---

## 12. 附录：关键文件索引

### DeepGEMM-jasl（`sm120` 分支）
```
deep_gemm/include/deep_gemm/mma/sm120.cuh                            # SM120 MMA 原语
deep_gemm/include/deep_gemm/impls/sm120_fp8_einsum.cuh                # FP8 einsum
deep_gemm/include/deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh      # FP8 paged MQA 打分
deep_gemm/include/deep_gemm/impls/sm120_tf32_hc_prenorm_gemm.cuh      # HC prenorm GEMM
csrc/jit_kernels/heuristics/sm120.hpp                                 # JIT 派发启发式
tests/test_sm120_kernels.py                                           # 回归测试
```

### vLLM（`ds4-sm120-preview-dev`）
```
vllm/v1/attention/backends/mla/sparse_mla_kernels.py   (3521 行)      # 可移植 Triton sparse MLA
vllm/v1/attention/backends/mla/sparse_mla_env.py                      # SM12x 自动启用 / 旋钮
vllm/models/deepseek_v4/nvidia/flashmla.py             (827 行)        # sparse MLA 前向 + 派发
vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py (706)  # DeepGEMM 接口回退
vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py        (726 行)        # MQA Triton / top-k
vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py                       # FP8 einsum 入口
vllm/model_executor/warmup/kernel_warmup.py            (424 行)        # 启动预热
vllm/reasoning/deepseek_v4_reasoning_parser.py         (304 行)        # 长上下文 reasoning 兜底
vllm/v1/core/sched/scheduler.py                                       # 长 prefill 公平性
vllm/utils/deep_gemm.py                                               # DeepGEMM 导入/检测
vllm/envs.py                                                          # 环境变量定义
```

### 配置 JSON（SM12x 手调，节选）
```
vllm/model_executor/layers/quantization/utils/configs/N=*,K=*,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,...json
vllm/model_executor/layers/quantization/utils/configs/N=*,K=*,device_name=NVIDIA_GB10,...json
vllm/model_executor/layers/fused_moe/configs/E=*,N=*,device_name=NVIDIA_RTX_PRO_6000_*...json
```
