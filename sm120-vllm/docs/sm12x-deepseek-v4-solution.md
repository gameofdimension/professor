# DeepSeek V4 在 SM12x 上的 vLLM 推理方案

> 作者：jasl（DeepSeek V4 SM12x 后端移植与优化）
> 涉及仓库：`vllm-project/vllm`（分支 `ds4-sm120-preview-dev`）+ 配套 `DeepGEMM-jasl`（分支 `sm120`）
> 整理日期：2026-06-16（第 8 节及关联章节已于 2026-06-16 订正）

本文档梳理 jasl 为 **DeepSeek V4 在 NVIDIA Blackwell SM12x GPU 上的 vLLM 推理支持** 所做的完整工程方案：根因、总体架构、两个仓库的关系、一次前向的数据流、**DeepGEMM 在运行时的真实角色**、分支模型、使用方式与完成度评估。

> **阅读提示**：经逐行追踪派发代码，本方案有一个反直觉但关键的事实——**vLLM 的 SM12x 运行路径是自包含的，运行时根本不调用 DeepGEMM-jasl fork 的任何计算内核**。DeepGEMM-jasl 是作者 4 月的可行性验证产物，5 月集成进 vLLM 时被自研 SM12x Triton/torch 取代。详见第 8 节。

---

## 1. 一句话概述

把 DeepSeek V4 的推理支持从**数据中心 Blackwell（SM90/SM100，B200）**下沉到**消费/工作站/边缘 Blackwell（SM12x）**。作者先后做了两件事：

1. **DeepGEMM-jasl fork（4 月）**：给 DeepGEMM 补齐 SM12x 的张量核心原语，证明 DeepGEMM 的 CUDA 路线在 SM120 上可跑（解除"stock DeepGEMM 拒绝 SM12x"的硬卡点）。
2. **vLLM SM12x 集成（5–6 月）**：在 vLLM 里用**自包含的 SM12x Triton/torch**实现所有原本依赖 DeepGEMM 的算子，外加可移植 sparse MLA、手调 FP8 配置、调度器加固等，把完整前向与服务化跑通。

> 注意：第 2 步并未复用第 1 步的 fork 内核——vLLM 在 wrapper 层把所有 DeepGEMM 接口短路到了自己的 SM12x 实现。**fork 对 vLLM 运行时非必需**（详见 §8）。

## 2. 目标硬件

| 架构代号 | 设备 | 用途 |
|---|---|---|
| **SM120**（GB202） | RTX PRO 6000 Blackwell Workstation Edition | 工作站 |
| **SM121**（Grace-Blackwell） | GB10（DGX Spark / Jetson Thor） | 边缘/嵌入式 |

关键基础工作（在 fork 里）：把 CUDA major 12（SM120 + SM121）统一归一到 **`sm_120f` JIT target + `sm120` include 后缀**，使这两类设备共享同一套 SM12x 内核族。

## 3. 根因：为什么 SM12x 开箱跑不了 DSv4

DeepSeek V4 的前向路径依赖两类上游库，而**它们都只支持 SM90/SM100**：

1. **DeepGEMM** —— 提供 FP8/FP4 GEMM、fused MoE（Mega MoE）、paged MQA logits（闪电 indexer 打分）、HyperConnection(HC) prenorm GEMM 等张量核心原语。上游 README 明确写 *"NVIDIA SM90 or SM100 architecture GPU"*，**SM12x 设备会在 dispatch 时被直接拒绝**。
2. **FlashInfer / CUTLASS sparse MLA** —— 提供完整 sparse MLA 注意力，在 SM12x 上不可用或稳定。

DeepGEMM-jasl 的提交 `2206a1d` 把根因讲得很直白：

> *"DeepGEMM currently rejects SM12x devices in the DeepSeek V4 forward path because the HC prenorm GEMM and paged MQA logits families only dispatch SM90/SM100 implementations."*

因此 SM12x（RTX PRO 6000 Blackwell、GB10/DGX Spark）上 DSv4 根本无法启动，更谈不上服务化。

## 4. 总体架构：两个仓库、两条并行路线

```
┌─────────────────────────────────────────────────────────────────┐
│ vLLM  ds4-sm120-preview-dev   （SM12x 自包含服务层 —— 实际运行路径）│
│   • 可移植 Triton sparse MLA 全注意力                              │
│   • 所有 DeepGEMM-interface 算子的 SM12x Triton/torch 实现         │
│     einsum → fp8_einsum.py 的 SM12x Triton                         │
│     mqa/paged-mqa/hc → sm12x_deep_gemm_fallbacks（Triton/torch）   │
│   • 通用 FP8 GEMM/MoE → CUTLASS(SM120)/Marlin/Triton/FlashInfer   │
│   • 手调 FP8 配置 + warmup + 调度器公平性 + MTP + reasoning parser │
└─────────────────────────────────────────────────────────────────┘
              ▲ SM12x 运行时【不】向下调用 fork 的计算内核
              │（wrapper 用 is_device_capability_family(120) 短路）

┌─────────────────────────────────────────────────────────────────┐
│ DeepGEMM-jasl  sm120   （独立的 CUDA 能力 / 参考实现，4 月产物）   │
│   • sm120 FP8 einsum / paged MQA logits / tf32 HC prenorm GEMM   │
│   • SM120 MMA 原语（避开 SM90 WGMMA / SM100 tcgen05+TMA 假设）     │
│   • SM12x JIT 派发启发式、sm_120f 统一 target                      │
│   ⚠️ vLLM 的 SM12x 派发不会调用这些内核（见 §8）                    │
└─────────────────────────────────────────────────────────────────┘
```

**设计哲学**：vLLM 在 SM12x 上**不依赖 DeepGEMM-jasl**——每个 DeepGEMM 接口都用 SM12x 自研 Triton/torch 实现，并在 wrapper 层短路，使方案对"是否安装 fork"完全鲁棒。DeepGEMM-jasl 则作为**独立的 DeepGEMM 能力 / 正确性与性能基准**存在，可在 vLLM 之外的 SM12x 推理栈直接使用。

---

## 5. Repo 1：DeepGEMM-jasl `sm120` 分支（CUDA 原语层）

- **时间**：2026-04-24 ~ 04-26（**先于 vLLM 侧**，作为可行性验证）
- **规模**：14 个提交（13 jasl + 1 ergodic-flow），约 **+2,035 行**
- **定位**：扩展 DeepGEMM 本体，证明其 CUDA 路线可在 SM12x 跑通。**注意：这些内核不在 vLLM 的 SM12x 运行路径上（见 §8）**，作为独立 DeepGEMM 能力 / 参考实现 / 基准存在。

### 5.1 核心改动

| 文件 | 作用 |
|---|---|
| `deep_gemm/include/deep_gemm/mma/sm120.cuh`（62 行） | SM120 矩阵乘累加原语，避开 SM90 WGMMA / SM100 tcgen05+TMA 的硬件假设 |
| `deep_gemm/.../impls/sm120_fp8_einsum.cuh`（126 行） | DSv4 O-proj 等 FP8 einsum |
| `deep_gemm/.../impls/sm120_fp8_paged_mqa_logits.cuh`（395 行） | sparse MLA indexer 的 FP8 paged MQA 打分（直接 dequant → FP32 累加 → ReLU → per-head weight，无效位置写 -inf） |
| `deep_gemm/.../impls/sm120_tf32_hc_prenorm_gemm.cuh`（288 行） | HyperConnection prenorm GEMM（保留 split-output 带 square-sum 的 ABI） |
| `csrc/jit_kernels/heuristics/sm120.hpp`（243 行） | SM12x JIT 内核派发启发式 |

### 5.2 基础设施改动

- **SM120/SM121 归一**：CUDA major 12 → `sm_120f` JIT target + `sm120` include 后缀（统一内核族）。`device_runtime.hpp` 的 `get_arch()` 把 major==12 归一为 `"120"`/`"120f"`。
- **CUTLASS 升级**到 4.4.2。
- **参考回退（reference fallbacks）**：早期用 ATen CUDA matmul 提供 HC prenorm 等的正确性回退，后续替换为真内核。
- **回归测试**：`tests/test_sm120_kernels.py`（380 行）覆盖 HC prenorm GEMM 与 FP8 paged MQA logits（含 fused paged KV-cache 布局）。

### 5.3 "gating" 机制：未移植算子在 SM12x 上如何被拒

DeepGEMM 每个算子 API 用 `device_runtime->get_arch_major()` 做架构派发梯子，**未移植到 SM12x 的算子没有 `arch_major == 12` 这一档，落到 `DG_HOST_UNREACHABLE(...)`**（展开为 `throw DGException`，会冒泡成 Python 端 RuntimeError）。举例（均在 fork 的 `csrc/apis/`）：

- 通用 FP8/FP4 GEMM（`gemm.hpp`）：`if(9) sm90 else if(10) sm100 else UNREACHABLE` → SM12x 被 gate。
- 非分页 MQA logits（`attention.hpp`）：无 12 分支 → SM12x 被 gate（FP4 与非分页 FP8 都没移植）。
- 分页 MQA logits（`attention.hpp`）：**只有非-FP4(FP8) 有 12 分支**（`sm120_fp8_paged_mqa_logits`），FP4 版仍 gate。
- MegaMoE：无 12 分支 → 被 gate。

所谓 *"FP4 paged MQA, non-paged MQA, MegaMoE, and general FP8/FP4 GEMM remain gated"*，精确含义就是这些算子的派发梯子里没有 SM12x 分支、一旦在 SM12x 被调到就抛异常。vLLM 侧用 §8 的短路机制保证永远不会触发。

---

## 6. Repo 2：vLLM `ds4-sm120-preview-dev`（SM12x 服务层 —— 实际运行路径）

- **时间**：2026-05-05 ~ 06-16（持续开发）
- **规模**：101 个 jasl 提交（另 2 个外部协作者各 1 个），约 **+14,500 / -381 行**，跨 106 个文件
- **定位**：用自包含的 SM12x Triton/torch 实现所有 DeepGEMM-interface 算子 + 可移植 sparse MLA + 服务化加固。**不依赖 fork 内核。**

### 6.1 工作线一览

| 工作线 | 关键文件 | 内容 |
|---|---|---|
| **可移植 Triton sparse MLA**（核心） | `vllm/v1/attention/backends/mla/sparse_mla_kernels.py`（3521 行）、`sparse_mla_env.py` | 重写 sparse MLA 全流程：decode + prefill + D512 split/chunked prefill；SM12x 自动启用 |
| **DeepGEMM 接口的 SM12x 实现** | `vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py`（706 行）、`sm12x_mqa.py`（726 行）、`fp8_einsum.py` | 用 Triton/torch 实现 mqa logits / paged mqa logits / hc prenorm / einsum，在 wrapper 层短路掉 DeepGEMM |
| **手调 FP8 配置** | `vllm/model_executor/layers/quantization/utils/configs/*.json` 等 ~25 个 | 为 RTX PRO 6000 / GB10 调 dense GEMM 与 fused-MoE 的 FP8 W8A8 block tile |
| **FP4 MoE** | `mxfp4.py`、`nvfp4.py`、`routed_experts.py`、`flashinfer_cutlass_moe.py` | NVFP4 ModelOpt 路由、FlashInfer CUTLASS MXFP4 opt-in、MoE metadata/EPLB |
| **kernel warmup 基建** | `vllm/model_executor/warmup/kernel_warmup.py`（424 行） | 启动时预热 JIT Triton 内核，避免热路径 JIT |
| **调度器长 prefill 公平性** | `vllm/v1/core/sched/scheduler.py`（+153 行）、测试 +578 行 | 超长 prefill 不饿死在跑的 decode/prefill/缓存长 prompt 尾部；prefix-cache 块哈希修复 |
| **MTP spec decode** | `nvidia/mtp.py`、`vllm/v1/spec_decode/llm_base_proposer.py` | MTP 调度稳定化、workspace warmup、sparse SWA 重排、小批 cudagraph 挂起修复 |
| **服务化加固** | `vllm/reasoning/deepseek_v4_reasoning_parser.py`（304 行） | 长上下文（~95k–100k）下 DSv4 偶尔漏 `</think>` 的防御式补全、DSML tool-call 流式标记处理 |
| **加载与配置** | `weight_utils.py`、`config/vllm.py`、`config/compilation.py` | GB10 加载开销优化、权重过滤快速 safetensors、SM121 跳过 cudagraph 自动启用 |
| **持续 rebase 跟随上游** | 多处 | 上游 DSv4 在重构（MegaMoE、runner refactor、metadata split、#45061），大量 "after rebase" 清理 |

### 6.2 sparse MLA 的派发逻辑

`vllm/models/deepseek_v4/nvidia/flashmla.py` 中，`is_triton_sparse_mla_enabled(q.device)`（`flashmla.py:803/947`）控制是否走 Triton 路径。SM12x 上自动为 True（`sparse_mla_env.py` 的 `_is_sm12x_device` 检测 capability major==12）。

---

## 7. 一次 DSv4 前向在 SM12x 上的数据流

**全部走 vLLM 自实现，不经过 DeepGEMM 计算内核。**

```
DSv4 forward on SM12x
│
├─ Dense FP8 GEMM（QKV / dense 层）
│    └─ CUTLASS scaled_mm(SM120) / Marlin（is_deep_gemm_supported()=False → 不选 deep_gemm）
│
├─ O-proj FP8 einsum (bhr,hdr->bhd)
│    └─ vLLM SM12x Triton  deepseek_v4_sm12x_fp8_einsum（fp8_einsum.py）
│
├─ MoE 专家
│    └─ fused-MoE Triton / Marlin / FlashInfer CUTLASS / ModelOpt NVFP4
│       （DeepGEMM MegaMoE 在 SM12x 不在路径）
│
├─ Sparse MLA 注意力（核心）
│   ├─ 分页/非分页 MQA logits ─► vLLM sm12x_deep_gemm_fallbacks（Triton/torch）
│   ├─ HC prenorm GEMM ─► vLLM sm12x_deep_gemm_fallbacks（torch）
│   ├─ row top-k（免物化全 logits）─► vLLM sm12x_mqa.py Triton 分块 top-k
│   └─ 稀疏注意力（gather KV / softmax / merge）─► vLLM sparse_mla_kernels.py（Triton）
│
├─ MTP spec decode ─► vLLM proposer + warmup
│
└─ 全程：启动 kernel warmup 预热 JIT；运行期 调度器公平性 + prefix cache
```

---

## 8. DeepGEMM 的运行时角色（订正版）

> ⚠️ 本节订正了早期版本的错误认识。早期误以为"装了 DeepGEMM-jasl fork → vLLM 在 SM12x 走 fork 的 CUDA 满血内核"。逐行追踪派发代码后确认：**vLLM 的 SM12x 运行路径是自包含的，不调用 fork 的任何计算内核。**

### 8.1 SM12x 上每个 DeepGEMM 接口的实际走向

下表"证据"列：前 5 行与后 3 行均在 **vLLM repo**（`/root/vllm-jasl`）。

| DSv4 算子 | vLLM 入口 | SM12x 实际路径 | 证据（行号，vLLM repo） |
|---|---|---|---|
| O-proj FP8 einsum `bhr,hdr->bhd` | `deepseek_v4_fp8_einsum` | **vLLM SM12x Triton** `deepseek_v4_sm12x_fp8_einsum` | `vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py:186-199,269-271` |
| 非分页 MQA logits（FP8） | `fp8_fp4_mqa_logits` | **vLLM 回退** `_fp8_mqa_logits_sm12x` | `vllm/utils/deep_gemm.py:452-455` |
| 分页 MQA logits（FP8） | `fp8_fp4_paged_mqa_logits` | **vLLM 回退** `_fp8_paged_mqa_logits_sm12x` | `vllm/utils/deep_gemm.py:573-576` |
| MQA top-k（免物化 logits） | `fp8_fp4_mqa_topk_indices` / `..._paged_...` | **vLLM 回退**（SM12x 专用，否则 return False） | `vllm/utils/deep_gemm.py:378-402,505-531` |
| HC prenorm GEMM | `tf32_hc_prenorm_gemm` | **vLLM 回退** `_tf32_hc_prenorm_gemm_sm12x` | `vllm/utils/deep_gemm.py:620-621` |
| 通用 FP8 dense GEMM | `scaled_mm` 后端选择 | **非 DeepGEMM**（CUTLASS SM120 / Marlin） | `vllm/model_executor/kernels/linear/scaled_mm/deep_gemm.py:50`；`vllm/platforms/cuda.py:559`（`support_deep_gemm` 不含 SM12x） |
| FP8/FP4 MoE | `fused_moe` 后端选择 | **非 DeepGEMM**（Triton/Marlin/FlashInfer/ModelOpt） | 同上（`is_deep_gemm_supported()=False`） |
| MegaMoE | `fp8_fp4_mega_moe` | **不在 SM12x 路径**（DeepGEMM 内 gated + vLLM 不选） | `vllm/utils/deep_gemm.py:371`、`vllm/models/deepseek_v4/nvidia/model.py:453` |

### 8.2 对应到 fork 的三个 SM120 CUDA 内核

| Fork 内核（DeepGEMM-jasl repo） | 行数 | vLLM SM12x 是否调用 |
|---|---|---|
| `sm120_fp8_einsum.cuh` | 126 | ❌ 否（O-proj 用 vLLM Triton，`fp8_einsum.py:269`） |
| `sm120_fp8_paged_mqa_logits.cuh` | 395 | ❌ 否（wrapper 短路到 vLLM 回退，`deep_gemm.py:573`） |
| `sm120_tf32_hc_prenorm_gemm.cuh` | 288 | ❌ 否（wrapper 无条件短路到 vLLM 回退，`deep_gemm.py:620`） |

**机制（两层咬合）**：

1. **wrapper 层短路**（`vllm/utils/deep_gemm.py`）：每个 DeepGEMM 接口 wrapper 在最前面用 `current_platform.is_device_capability_family(120)` 判断，命中即返回 vLLM 自研实现，根本到不了 DeepGEMM 的 `_*_impl`。
2. **平台层排除**（`vllm/platforms/cuda.py:559`）：`support_deep_gemm()` 只认 `is_device_capability(90) or is_device_capability_family(100)`，**不含 SM12x** → `is_deep_gemm_supported()` 在 SM12x 为 False → 通用 GEMM/MoE 的 `scaled_mm`/`fused_moe` 后端选择在 SM12x 上根本不考虑 DeepGEMM（`scaled_mm/deep_gemm.py:50` 直接 return unsupported）。

> 唯一可能仍触达 DeepGEMM 的是几个**布局/缩放辅助函数**（`transform_sf_into_required_layout` 等，见 `fp8_utils.py:1143/1166`），但那只是 scale 张量重排，非性能内核，也非 fork 价值所在；且 SM12x 主路径不依赖其产出。

### 8.3 结论

- **vLLM 的 SM12x DSv4 路径自包含，运行时不依赖 DeepGEMM-jasl 的任何计算内核。** 装不装 fork，SM12x 都走 vLLM 自研 Triton/torch。
- 因此"去掉 DeepGEMM-jasl，DSv4 在 SM12x 照样跑"不只是"有替代后端兜底"——而是**主路径本就不经过 fork**。
- **DeepGEMM-jasl fork 的真实定位**：jasl **4 月**的可行性验证（证明 DeepGEMM CUDA 路线在 SM120 上可跑：避开 SM90 WGMMA / SM100 tcgen05+TMA、自写 SM120 MMA）；**5 月**集成进 vLLM 时，他转而用自包含 SM12x Triton/torch，并在 wrapper 层短路掉 DeepGEMM。fork 遂成为**独立的 DeepGEMM 能力 / 正确性与性能基准 / 参考实现**，可在 vLLM 之外的 SM12x 推理栈直接使用，但不作为 vLLM 运行时必需项。

> **关于"为什么还做了 fork"**：这是同一个人的两次尝试——4 月先用 CUDA（fork）证明可行，5 月在 vLLM 集成时改用自包含 Triton/torch（更易维护、不背 fork 依赖、Triton 在这些 shape 上够用），后者在 vLLM 内覆盖了前者的作用。fork 保留了独立的 DeepGEMM-SM12x 能力与基准价值。

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
04-24 ~ 04-26  DeepGEMM-jasl sm120 分支（CUDA 可行性验证：SM12x 原语 + gating）
05-05 ~ 05-07  vLLM 侧奠基：portable sparse MLA Triton + SM12x 自包含算子 + 串接前向
05-08 ~ 05-20  内核调优、warmup、MQA top-k、FP8 配置
05-21 ~ 06-16  调度器公平性、FP4 MoE、稳定化、rebase 跟随、清理（持续）
```

### 9.3 preview 独有但 dev 中未见的项（需确认）

- **`Support DeepSeek V4 pipeline parallel preview`**（Isotr0py，5/3）——在 dev 代码中按 `pipeline_parallel` 关键词无命中。dev 里 `deepseek_v4` 整个被重构（旧单文件 → 新包），PP 可能被吸收或暂未迁移。**如需 PP 建议单独确认。**
- `Stabilize SM12x piecewise CUDA graphs` + 其 Revert（在 preview tip 相互抵消，净效果未启用）——dev 走了另一条路（SM121 跳过 cudagraph 自动启用等）。

---

## 10. 使用方式

### 10.1 核心设计：零配置自动启用

在 SM12x（capability major == 12）上，sparse MLA 的 Triton 路径与相关开关**默认自动启用**，通常无需手动设置。**vLLM 的 SM12x 路径自包含，不需要安装 DeepGEMM-jasl**——DeepGEMM-jasl 仅当需要独立的 DeepGEMM-SM12x 能力/基准时才安装（`vllm/utils/deep_gemm.py` 的 `_import_deep_gemm` 会优先用 site-packages 的 deep_gemm，但 SM12x 主路径不依赖它）。

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

- **无面向用户的 SM12x 专门文档/README/example**。jasl 在本分支从未碰 `docs/`（本文档除外）。
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

### DeepGEMM-jasl（`sm120` 分支，`/root/DeepGEMM-jasl`）
```
deep_gemm/include/deep_gemm/mma/sm120.cuh                            # SM120 MMA 原语
deep_gemm/include/deep_gemm/impls/sm120_fp8_einsum.cuh                # FP8 einsum（vLLM 不调用）
deep_gemm/include/deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh      # FP8 paged MQA 打分（vLLM 不调用）
deep_gemm/include/deep_gemm/impls/sm120_tf32_hc_prenorm_gemm.cuh      # HC prenorm GEMM（vLLM 不调用）
csrc/jit_kernels/heuristics/sm120.hpp                                 # JIT 派发启发式
csrc/apis/{gemm,attention,mega,hyperconnection}.hpp                   # 架构派发梯子 + gating
csrc/jit/device_runtime.hpp                                           # get_arch / sm_120f 归一
tests/test_sm120_kernels.py                                           # 回归测试
```

### vLLM（`ds4-sm120-preview-dev`，`/root/vllm-jasl`）—— 实际运行路径
```
vllm/v1/attention/backends/mla/sparse_mla_kernels.py   (3521 行)      # 可移植 Triton sparse MLA
vllm/v1/attention/backends/mla/sparse_mla_env.py                      # SM12x 自动启用 / 旋钮
vllm/models/deepseek_v4/nvidia/flashmla.py             (827 行)        # sparse MLA 前向 + 派发
vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py (706)  # mqa/paged-mqa/hc 的 SM12x 实现
vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py        (726 行)        # MQA Triton / top-k
vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py                       # O-proj SM12x Triton einsum
vllm/utils/deep_gemm.py                                               # DeepGEMM wrapper + SM12x 短路
vllm/platforms/cuda.py:557-559                                         # support_deep_gemm (不含 SM12x)
vllm/model_executor/kernels/linear/scaled_mm/deep_gemm.py              # FP8 GEMM 后端（SM12x 不选）
vllm/model_executor/warmup/kernel_warmup.py            (424 行)        # 启动预热
vllm/reasoning/deepseek_v4_reasoning_parser.py         (304 行)        # 长上下文 reasoning 兜底
vllm/v1/core/sched/scheduler.py                                       # 长 prefill 公平性
vllm/envs.py                                                          # 环境变量定义
```

### 配置 JSON（SM12x 手调，节选）
```
vllm/model_executor/layers/quantization/utils/configs/N=*,K=*,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,...json
vllm/model_executor/layers/quantization/utils/configs/N=*,K=*,device_name=NVIDIA_GB10,...json
vllm/model_executor/layers/fused_moe/configs/E=*,N=*,device_name=NVIDIA_RTX_PRO_6000_*...json
```
