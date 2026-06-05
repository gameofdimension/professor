<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SID-GR Inference 深度剖析

> 面向"长 context + 短 decode + 大 beam width"推荐/搜索/广告场景的专用生成式推理框架。
> 本文档从动机、设计、效果、实现细节四个维度，按模块进行完整剖析。

> 项目地址：https://github.com/NVIDIA/recsys-examples/blob/2091502f442a0070668bcc6f35913fd42bacc444/examples/sid-gr-inference/README.md?plain=1#L6
---

## 目录

- [一、动机](#一动机)
- [二、设计](#二设计)
  - [2.1 架构总览](#21-架构总览)
  - [2.2 与通用 LLM Serving 的设计对比](#22-与通用-llm-serving-的设计对比)
- [三、效果](#三效果)
  - [3.1 离线性能](#31-离线性能)
  - [3.2 在线 Serving](#32-在线-serving)
  - [3.3 性能根因](#33-性能根因nsight-breakdown)
  - [3.4 正确性](#34-正确性)
- [四、模块剖析](#四模块剖析)
  - [4.1 gr_kv/ — SID-GR 原生 KV 抽象层](#41-gr_kv--sid-gr-原生-kv-抽象层)
  - [4.2 gr_kernels/ — Kernel 后端注册与调度](#42-gr_kernels--kernel-后端注册与调度)
  - [4.3 gr_models/ — Qwen3 模型集成](#43-gr_models--qwen3-模型集成)
  - [4.4 gr_runtime/ — Beam Search 运行时](#44-gr_runtime--beam-search-运行时)
  - [4.5 gr_serving/ — 连续批调度与 HTTP Serving](#45-gr_serving--连续批调度与-http-serving)
- [五、端到端交互全流程](#五端到端交互全流程)
- [六、关键设计决策总结](#六关键设计决策总结)

---

## 一、动机

### 1.1 业务背景

Semantic ID-based Generative Recommender (SID-GR) 是推荐/搜索/广告系统的一个主要方向。核心工作流：

```
离线聚类 → 将每个真实 item ID 映射为多层 cluster ID 元组
         → 推理时自回归地生成一个短的 semantic ID 序列
         → 将生成的 semantic ID 元组映射回真实 item ID
```

### 1.2 核心推理模式

这种工作负载形成了一个独特的推理模式：**"长 context + 短 decode + 大 beam width"**

| 特征 | 说明 |
|---|---|
| **长 context** | 用户历史行为可达 1k–5k tokens，prefill 计算成本占主导 |
| **短 decode** | 实际生成通常只有 3–5 步（cluster 深度很小） |
| **大 beam** | beam width 通常为 128 或 256，以获取推荐多样性 |

### 1.3 为什么不用通用 LLM Serving 框架

| 框架 | 问题 |
|---|---|
| **vLLM** | 不提供稳定的 beam-search serving 路径，用户通常需要从业务逻辑反复调用 vLLM |
| **TensorRT-LLM** | 不原生暴露 logprobs，大 beam width 造成内存压力，decode attention kernel 未针对短 SID-GR decode 优化 |
| **SGLang** | 最可用的开源基线，但 large-beam 支持仍在 feature PR 中，未合入上游 |

通用框架将 `batch * beam` 展开为大量独立的 decode rows，无法利用 **同一 request 内所有 beam 共享 context** 的结构特性。

### 1.4 目标

- 优化 Qwen-family 模型向 SID-GR workload 的速度极限：长 context、短 decode、大 beam width。
- 提供紧凑框架，同时满足 SID-GR serving 的功能需求和性能需求。

---

## 二、设计

### 2.1 架构总览

```
┌─────────────────────────────────────────────────────────┐
│  gr_serving/                                            │
│  HTTP /generate, continuous batching, CUDA graph,       │
│  memory pools, metrics, prefix cache                    │
├─────────────────────────────────────────────────────────┤
│  gr_runtime/                                            │
│  Beam search decode loop, logits processing,            │
│  beam policies, item constraints, timing                │
├─────────────────────────────────────────────────────────┤
│  gr_models/                                             │
│  Qwen3 model integration, weight loading, layer ops     │
├─────────────────────────────────────────────────────────┤
│  gr_kernels/                                            │
│  Kernel backend registry, decode/prefill attention      │
│  dispatch, TRT-LLM experimental CUDA kernels            │
├─────────────────────────────────────────────────────────┤
│  gr_kv/                                                 │
│  ContextKV, BeamKV, BeamPath — SID-GR 原生数据结构      │
└─────────────────────────────────────────────────────────┘
```

**设计哲学：** 保持 SID-GR 原生抽象，选择性复用成熟的 serving 技术。

### 2.2 与通用 LLM Serving 的设计对比

| 维度 | 通用 LLM Serving | SID-GR Inference |
|---|---|---|
| **KV 抽象** | Sequence/token block/paged KV | 显式 `ContextKV` + 短 `BeamKV` |
| **大 beam decode** | 将 `batch*beam` 展平为大量 decode rows | 按 request 和 beam tile 处理共享 context |
| **Attention kernel** | 通用 paged decode attention | `gr-decode_atten` 直接接收 `ContextKV + BeamKV + BeamPath` |
| **Batch CUDA graph** | 通用 batch bucket + paged KV 约束 | 固定 SID-GR shape + 稳定 pool slice replay |
| **输出** | 通用 API 输出 + beam 管理 | 快速路径返回 `beam_results`；调试路径可选 `beam_details` |

---

## 三、效果

> 测试环境：NVIDIA H100 80GB HBM3，Qwen3-1.7B，radix off / 无 prefix 复用。

### 3.1 离线性能

全部 24 个测试 case (ctx=1000/5000, batch=1/2/4/8, beam=256)，SID-GR **24/24 领先**。

| ctx | batch | SID-GR (ms) | SGLang (ms) | 加速比 |
|---:|---:|---:|---:|---:|
| 1000 | 1 | 17.6 | 33.5 | **1.90x** |
| 1000 | 4 | 47.7 | 102.3 | **2.14x** |
| 1000 | 8 | 93.2 | 199.3 | **2.14x** |
| 5000 | 1 | 42.3 | 94.6 | **2.24x** |
| 5000 | 4 | 154.2 | 349.9 | **2.27x** |
| 5000 | 8 | 307.9 | 685.4 | **2.23x** |

### 3.2 在线 Serving

固定 case: `ctx=5000, beam=256, output=3, requests=64, max_concurrency=4`

| Server | req/s | median ms | p99 ms |
|---|---:|---:|---:|
| **SID-GR** `/generate` | **~19.5** | **~199** | ~293 |
| SGLang beam | ~10.7 | ~370 | ~378 |

- 吞吐量约 **1.85x**，中位延迟约 **46% 降低**

### 3.3 性能根因（Nsight Breakdown）

以 `ctx=1000, beam=256, batch=4` 为例：

| 指标 | SID-GR | SGLang |
|---|---:|---:|
| Active CUDA window | 46.9 ms | 99.9 ms |
| **Decode attention kernels** | **1.6 ms** | **35.2 ms** |
| CUDA graph launches | 8 | 29 |
| CPU overhead gaps >50μs | 2.2 ms | 28.3 ms |

**核心差距在 decode attention：** SGLang 将 `4*256=1024` 个 decode row 各自走通用 paged attention 路径；SID-GR 保持 4 个 request 共享 ContextKV，256 beam 只 attend 短 BeamKV history。仅 decode attention 一项就差 **33.6 ms**。

### 3.4 正确性

所有 8 个固定 beam=256 case 的 Top1 完全一致 (exact agreement)，TopK overlap min 0.945 / mean 0.960。

---

## 四、模块剖析

### 4.1 `gr_kv/` — SID-GR 原生 KV 抽象层

> 这是整个框架的基石，定义了区别于通用 LLM serving 的核心数据结构。

#### 4.1.1 `layouts.py` — 无 GPU 依赖的形状描述

```python
Shape = tuple[int, ...]  # 纯 Python，测试不需要 torch
```

**核心类型：**

| 类型 | 作用 |
|---|---|
| `TensorSpec` (frozen dataclass) | `(name, shape, dtype, device, layout)` — 逻辑张量描述符 |
| `normalize_shape()` | 强制每个维度 > 0 |
| `shape_of(tensor)` | 兼容 `TensorSpec` 和真实 `torch.Tensor`（鸭子类型） |

`TensorSpec` 的 `__post_init__` 使用 `object.__setattr__` 在 frozen dataclass 内执行验证，这是 Python frozen dataclass 的标准模式。

**设计意图：** 测试层可以用 `TensorSpec` 跑全量校验逻辑，不需要 GPU。

#### 4.1.2 `context_kv.py` — 共享长 Context KV

```python
@dataclass(frozen=True)
class ContextKV:
    key: Any    # [layers, batch, context_len, kv_heads, head_dim]  (5D)
    value: Any
```

**关键行为：**

| 方法/属性 | 行为 |
|---|---|
| `slice_batch(index)` | 切片 `[:, index:index+1, ...]`，**保持 batch=1 不 squeeze**，作为单请求视图 |
| `expected_layer_shape()` | `(batch, context_len, kv_heads, head_dim)` — 每层视图 |
| `num_layers`, `batch_size`, `context_len`, `num_kv_heads`, `head_dim` | 按 shape 索引分解 |

**不变量：** `key.shape == value.shape`，且必须 5D。

**与通用框架的对比：** vLLM/SGLang 用 paged KV + block table；SID-GR 直接用密集连续的 5D 张量，因为同一 request 所有 beam 共享同一个 context。

#### 4.1.3 `beam_kv.py` — 短 Decode KV（step-major 布局）

```python
@dataclass(frozen=True)
class BeamKV:
    key: Any    # [layers, batch, max_decode_steps, max_beam_width, kv_heads, head_dim]  (6D)
    value: Any
```

**关键方法：**

| 方法 | 行为 |
|---|---|
| `validate_step(step, width)` | 校验 `0 ≤ step < max_decode_steps` 且 `0 < width ≤ max_beam_width`，返回 `BeamStepWrite` |
| `flattened_beam_shape()` | `(batch, max_decode_steps * max_beam_width, kv_heads, head_dim)` — kernel 友好视图 |
| `BeamStepWrite.flat_offset` | `= step * max_beam_width` — 在展平视图中的偏移量 |

**为什么 step 在 beam 之前？** 每个 decode step 的所有 beam 在内存中连续，写入时无需跨步访问。

**预分配与活跃分离：** 存储按 `max_decode_steps × max_beam_width` 预分配，但每步只写入 `active_beam_width` 个 slot。

```python
@dataclass(frozen=True)
class BeamStepWrite:
    step: int
    active_beam_width: int
    expected_step_shape: Shape   # (batch, active_beam_width, kv_heads, head_dim)
    flat_offset: int             # step * max_beam_width
```

#### 4.1.4 `beam_path.py` — Beam 祖先追踪

```python
@dataclass(frozen=True)
class BeamPathEntry:
    parent_beams: tuple[int, ...]    # 当前 step 每个 beam 的父 beam 索引
    token_ids:   tuple[int, ...]     # 每个 beam 选择的 token
    scores:      tuple[float, ...]   # 每个 beam 的累计分数
    width: int  # = len(parent_beams)

@dataclass
class BeamPath:                      # mutable — append-only log
    max_decode_steps: int
    max_beam_width: int
    entries: list[BeamPathEntry]
```

**核心算法 — `token_trace(beam, step)`：**

```
从 step 向前回溯到 step=0，每步走 parent_beams[current_beam]
收集 token_ids，反转后返回 root→leaf 的完整 token 序列
时间复杂度 O(step)
```

**验证规则：**
- 第一步的 `parent_beams` 必须全为 0（只有一个 root beam）
- 每步的 `parent_beams[i] < 前一步的 width`
- beam width 可增可减，但不超过 `max_beam_width`

**设计选择：** `tuple` 用于不可变的 `BeamPathEntry` 字段，防止历史路径数据意外变异；`BeamPath` 本身是 mutable 的 append-only log，O(1) 追加。

#### 4.1.5 `batched_beam_path.py` — 批量 Beam Path

```python
@dataclass
class BatchedBeamPath:
    paths: tuple[BeamPath, ...]    # batch_size 个独立的 BeamPath
```

- 通过 `TYPE_CHECKING` 守卫避免与 `gr_runtime` 循环导入
- `append(selection)` 通过 `zip` 将 `BatchedBeamSelection` 分发到各 `BeamPath`

```python
@dataclass
class BatchedBeamPathBuilder:    # 便利包装
    batch_size: int
    max_decode_steps: int
    max_beam_width: int
    # __post_init__ 创建 BatchedBeamPath
    # append(selection) -> BatchedBeamPath
```

#### 4.1.6 模块间依赖

```
layouts.py ←── context_kv.py
           ←── beam_kv.py
                          ←── beam_path.py
                                       ←── batched_beam_path.py ←── (BatchedBeamSelection via TYPE_CHECKING)
```

#### 4.1.7 核心设计模式

| 模式 | 应用 |
|---|---|
| **Frozen dataclass** | `TensorSpec`, `ContextKV`, `BeamKV`, `BeamStepWrite`, `BeamPathEntry` — 不可变值对象，安全共享 |
| **Mutable dataclass** | `BeamPath`, `BatchedBeamPath` — decode loop 中追加状态 |
| **鸭子类型存储** | `key: Any` + `shape_of()` — 兼容 TensorSpec 和真实 tensor |
| **`TYPE_CHECKING` 守卫** | 打破循环依赖，保留类型提示 |
| **两级 KV 分离** | ContextKV 共享 prompt KV，BeamKV 保存短 decode KV |

---

### 4.2 `gr_kernels/` — Kernel 后端注册与调度

> 能力导向的注册表 + 可配置优先级 + 环境变量覆盖 = 灵活的 kernel 调度。

#### 4.2.1 `registry.py` — 能力注册表

```python
@dataclass(frozen=True)
class KernelCapability:
    name: str  # 如 "rmsnorm", "rope", "gr_decode_attention"

@dataclass
class KernelBackendInfo:
    name: str
    capabilities: frozenset[str]
    available: bool = True
    metadata: dict[str, Any]
    def supports(self, capability: str) -> bool: ...

class KernelBackendRegistry:
    register(backend)             # 按 name 存储
    available_for(capability)     # 返回所有支持该能力的 backend（保序）
    prefer(capability, order)     # 按 order 优先级选第一个
    summary()                     # 诊断快照
```

**10 个已注册能力：**

| 常量 | 值 | 用途 |
|---|---|---|
| `CAP_RMSNORM` | `"rmsnorm"` | RMS 归一化 |
| `CAP_FUSED_ADD_RMSNORM` | `"fused_add_rmsnorm"` | 融合残差加 + RMSNorm |
| `CAP_ROPE` | `"rope"` | 旋转位置编码 |
| `CAP_ROPE_WITH_CACHE` | `"rope_with_cache"` | 带 KV cache 交互的 RoPE |
| `CAP_QK_NORM_ROPE` | `"qk_norm_rope"` | 融合 Q/K RMSNorm + RoPE |
| `CAP_PREFILL_ATTENTION` | `"prefill_attention"` | Dense causal prefill attention |
| `CAP_GR_DECODE_ATTENTION` | `"gr_decode_attention"` | GR 专用 beam decode attention |
| `CAP_PACKED_GEMM` | `"packed_gemm"` | Packed GEMM (cuBLASLt) |
| `CAP_FUSED_MLP` | `"fused_mlp"` | 融合 gate/up SiLU down-projection |
| `CAP_SAMPLING_TOPK` | `"sampling_topk"` | Top-K 采样 |

#### 4.2.2 `backends.py` — 7 个内置 Backend

| Backend | 能力 | 可用条件 |
|---|---|---|
| `torch` | rmsnorm, rope, qk_norm_rope, prefill_attention, packed_gemm, fused_mlp | torch 可导入 |
| `sgl_kernel` | fused_mlp | `sgl_kernel.silu_and_mul` 存在 |
| `torch_compile` | fused_mlp | `torch.compile` 存在 |
| `trtllm_aligned` | qk_norm_rope, fused_mlp, packed_gemm | `torch.ops.gr_trtllm` 或 `torch.ops.trtllm` 下有对应 op |
| `flashinfer` | rmsnorm, fused_add_rmsnorm, rope, rope_with_cache, qk_norm_rope, sampling_topk | `flashinfer` 包存在 |
| `flash_attn` | prefill_attention | `flash_attn` 包存在 |
| `gr_decode_atten` | gr_decode_attention | 始终注册 |

**TRT-LLM 探测逻辑（最复杂）：**
1. 检查 `gr_inference_trtllm_kernels` 模块
2. 回退到 `tokenspeed-trtllm-kernel` 分发
3. 回退到 `tokenspeed_trtllm_kernel` 或 `tensorrt_llm` 模块
4. 探测 `torch.ops.gr_trtllm` 和 `torch.ops.trtllm` 命名空间
5. 能力按实际检测到的 op 动态构建

#### 4.2.3 `selection.py` — 优先级调度策略

```python
DEFAULT_CAPABILITY_ORDER = {
    "rmsnorm":            ("flashinfer", "torch"),
    "fused_add_rmsnorm":  ("flashinfer",),
    "rope":               ("flashinfer", "torch"),
    "qk_norm_rope":       ("flashinfer", "torch", "trtllm_aligned"),
    "prefill_attention":  ("flash_attn", "torch"),
    "gr_decode_attention":("gr_decode_atten",),         # 唯一选择
    "packed_gemm":        ("torch", "trtllm", "cutlass_triton"),
    "fused_mlp":          ("trtllm_aligned", "torch", "sgl_kernel", ...),
    "sampling_topk":      ("flashinfer", "torch"),
}
```

**调度链（三级回退）：**

```
1. 环境变量覆盖 GR_INFERENCE_KERNEL_<CAPABILITY>=<backend>
2. JSON Profile 文件 GR_INFERENCE_KERNEL_PROFILE=<path>
3. 默认优先级 DEFAULT_CAPABILITY_ORDER
```

**环境变量：**

| 变量 | 值 | 效果 |
|---|---|---|
| `GR_INFERENCE_KERNEL_PRESET` | `"auto"` / `"flashinfer"` / `"torch"` | `"torch"` 所有能力走 torch；其他用默认 |
| `GR_INFERENCE_KERNEL_<CAPABILITY>` | backend name | 单能力覆盖 |
| `GR_INFERENCE_KERNEL_PROFILE` | file path | 从 JSON 加载 profile |

**单例管理：** `_default_registry` 和 `_default_policy` 延迟初始化，`reset_default_kernel_selection_policy()` 用于测试重置。

#### 4.2.4 `profile.py` — Kernel Selection Profile

```python
@dataclass(frozen=True)
class KernelProfile:
    schema_version: int
    model: dict[str, Any]       # 模型元数据
    target: dict[str, Any]      # 硬件元数据
    selected: dict[str, str]    # capability → backend name
    benchmarks: dict[str, Any]  # 基准数据

    load(path) / save(path)     # JSON 序列化
```

#### 4.2.5 `fused_mlp.py` — Fused MLP 后端

```python
class TorchFusedMLPBackend:          # 默认 Python 回退
    def __call__(self, hidden_states, ops):
        gate, up = ops.gate_up(hidden_states)
        intermediate = ops.silu_mul(gate, up)
        return ops.down_proj_only(intermediate)

class FusedMLP:                      # 薄 dispatch 包装
    backend: Any
    def __call__(self, hidden_states, ops):
        return self.backend(hidden_states, ops)
```

#### 4.2.6 `attention/` — GR Decode Attention

**`GRDecodeAttentionInputs`** (frozen dataclass)：

```python
@dataclass(frozen=True)
class GRDecodeAttentionInputs:
    q:              Any    # [B, W_t, Hq, D]          4D query
    context_kv:     ContextKV                         # [L, B, T_ctx, Hkv, D]
    beam_kv:        BeamKV                            # [L, B, S, W, Hkv, D]
    beam_path:      BeamPath                          # 祖先追踪
    layer_idx:      int                               # 层索引
    step:           int                               # 当前 decode step
    active_beam_width: int
    topk_indices:   Any | None  # [B, Sq, Hq, max_decode_nums, W]  kernel 索引张量
    decode_nums:    int | None  # 默认 = step
    return_lse:     bool
    backend_name:   str
```

**校验矩阵（20+ 项）：**

| 校验项 | 约束 |
|---|---|
| Q rank | 4D `[B, W_t, Hq, D]` |
| batch 一致性 | `q.batch == context_kv.batch == beam_kv.batch` |
| beam width | `q.width == active_beam_width` |
| head_dim | `q.head_dim == context_kv.head_dim == beam_kv.head_dim` |
| layer 一致性 | `context_kv.num_layers == beam_kv.num_layers` |
| kv_heads | `context_kv.num_kv_heads == beam_kv.num_kv_heads` |
| layer_idx | `0 <= layer_idx < num_layers` |
| step | 委托给 `beam_kv.validate_step()` |
| BeamPath | `beam_path.steps_done <= step + 1` |
| topk_indices rank | 5D `[B, Sq, Hq, max_decode_nums, W]` |
| topk 维度 | batch/heads/decode_nums/beam_width 匹配 |

**`ExistingGRDecodeAttentionBackend`** — 外部 CuTe DSL kernel 适配器：

- 从 `third_party/gr-decode-attention/interface.py` 加载 `beam_decode_attn`
- Kernel root 解析优先级：构造参数 → `GR_DECODE_ATTEN_ROOT` 环境变量 → `third_party/gr-decode-attention`
- 张量变换：

| 变换 | 操作 |
|---|---|
| `_ensure_decode_q(q)` | `[B,W,Hq,D]` → unsqueeze → `[B,1,W,Hq,D]` |
| `_select_layer(tensor, idx)` | 5D → `tensor[idx]` → `[B,S,Hkv,D]` |
| `_select_beam_history(tensor, idx, nums, width)` | 6D → slice layer → reshape → `[B, nums*W, Hkv, D]` |

**外部 kernel 期望的 tensor 形状：**

```
q:            [B, 1, W, Hq, D]
k_context:    [B, S_ctx, Hkv, D]
v_context:    [B, S_ctx, Hkv, D]
k_beam:       [B, decode_nums * W, Hkv, D]
v_beam:       [B, decode_nums * W, Hkv, D]
topk_indices: [B, 1, Hq, max_decode_nums, W]
```

#### 4.2.7 `prefill/` — Prefill Attention

**三种后端：**

| Backend | 实现 |
|---|---|
| `TorchSDPAPrefillBackend` | `torch.nn.functional.scaled_dot_product_attention`；三种 GQA 路径：MHA 直接 / native GQA / repeat_interleave |
| `FlashAttentionPrefillBackend` | `flash_attn.flash_attn_func` |
| `SGLangFlashAttentionPrefillBackend` | `sglang.jit_kernel.flash_attention.flash_attn_varlen_func`；dense → varlen 转换 |
| `AutoPrefillBackend` | 首次调用时锁定最快的可用后端（lazy init） |

**`PrefillAttentionInputs` 和校验：**

```python
@dataclass(frozen=True)
class PrefillAttentionInputs:
    q: Any    # [B, S, Hq, D]
    k: Any    # [B, S, Hkv, D]
    v: Any    # [B, S, Hkv, D]
    layer_idx: int
    causal: bool = True
```

校验：Q/K rank=4D，K==V shape，batch/seq_len/head_dim 一致，GQA 可整除。

#### 4.2.8 `gr_inference_trtllm_kernels/` — 实验 CUDA Kernel（1513 行）

**6 个 JIT 编译的 CUDA kernel：**

| Kernel | 功能 | 线程配置 | 默认启用 |
|---|---|---|---|
| `fused_qk_norm_rope` | 融合 Q/K RMSNorm + RoPE | 256 线程，shared mem parallel reduction | ✅ `"1"` |
| `silu_and_mul_packed` | 向量化 SiLU×Mul | 16 字节 PackedVec vector load | ❌ `"0"` |
| `packed_gemm` | cuBLAS Tensor Core GEMM | cuBLAS GEMM | ❌ `"0"` |
| `exact_fused_add_rmsnorm` | 精确 add + RMSNorm 融合 | 256 线程，shared mem reduction | ✅ `"1"` |
| `write_beam_kv_step` | BeamKV 写入 (strided copy) | 256 线程，每元素每线程 | ❌ `"0"` |
| `write_packed_qkv_prefill_kv` | ContextKV 写入 (vectorized) | PackedVec vector load/store | ❌ `"0"` |

**每个 kernel 都有 Python reference fallback，通过 `_CALLS` 计数器跟踪调度路径。**

**Custom Op 注册（`torch.ops.gr_trtllm`）：**

```python
torch.library.Library("gr_trtllm", "DEF")
    .define("fused_qk_norm_rope(Tensor(a!) qkv, int, int, int, int, int, float, Tensor, Tensor, float, bool, Tensor, float, int, int, float, bool) -> ()")
    .define("gated_mlp(Tensor, Tensor, Tensor) -> Tensor")
    .define("packed_gemm(Tensor, Tensor, Tensor?) -> Tensor")
```

**CUDA kernel 技术细节：**

- `fused_qk_norm_rope_kernel`：`AT_DISPATCH_FLOATING_TYPES_AND2` 分发 fp16/bf16，`sincosf` 快速三角函数
- `silu_and_mul_packed_vec_kernel`：`PackedVec<scalar_t, VecSize>` 16 字节对齐向量读写
- `exact_fused_add_rmsnorm_kernel`：原地变异两个 tensor（CUDA graph 友好，无内部分配）
- `packed_gemm_cuda_launcher`：`cublasGemmEx` + `CUBLAS_COMPUTE_32F` + `CUBLAS_GEMM_DEFAULT_TENSOR_OP`

**环境变量门控：**

| 变量 | 默认 | 效果 |
|---|---|---|
| `GR_INFERENCE_GR_TRTLLM_KERNELS_JIT` | `"1"` | 总开关 |
| `GR_INFERENCE_GR_TRTLLM_BEAM_KV_WRITE_JIT` | `"0"` | BeamKV 写入 CUDA |
| `GR_INFERENCE_GR_TRTLLM_EXACT_ADD_RMSNORM_JIT` | `"1"` | 精确 add+norm CUDA |
| `GR_INFERENCE_GR_TRTLLM_GATED_MLP_JIT` | `"0"` | Gated MLP CUDA |
| `GR_INFERENCE_GR_TRTLLM_PACKED_GEMM_JIT` | `"0"` | Packed GEMM CUDA |
| `GR_INFERENCE_DEBUG_GR_TRTLLM` | unset | 打印调用计数 |

#### 4.2.9 模块间依赖图

```
gr_kernels/__init__.py
  → backends.py → package_imports.py, registry.py
  → registry.py (独立)
  → selection.py → backends.py, profile.py, registry.py
  → profile.py (独立)
  → fused_mlp.py (独立)

gr_kernels/attention/__init__.py
  → gr_decode_attention.py → gr_kv.layouts, gr_kv.beam_kv, gr_kv.beam_path, gr_kv.context_kv
  → existing_kernel_backend.py → gr_decode_attention.py

gr_kernels/prefill/__init__.py
  → base.py → gr_kv.layouts
  → auto_backend.py → base.py, flash_attn_backend.py, torch_sdpa_backend.py
  → flash_attn_backend.py → base.py
  → torch_sdpa_backend.py → base.py

gr_inference_trtllm_kernels/__init__.py
  → qwen3.py (独立，仅依赖 torch)
```

---

### 4.3 `gr_models/` — Qwen3 模型集成

#### 4.3.1 `config.py` — `Qwen3GRConfig`

```python
@dataclass(frozen=True)
class Qwen3GRConfig:
    model_name: str = "Qwen3-0.6B-GR"
    num_layers: int = 28
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_kv_heads: int = 8              # GQA
    head_dim: int = 128
    max_context_len: int = 4700
    max_seq_len: int = 4900
    max_decode_steps: int = 3
    max_beam_width: int = 128
    intermediate_size: int | None = None  # 默认 hidden_size * 4
    vocab_size: int | None = None
    tie_word_embeddings: bool = False
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0    # Qwen3 特有
    dtype: str = "bf16"
```

**计算属性：**

```
q_size = num_attention_heads * head_dim       # 2048 (0.6B) / 2048 (1.7B)
kv_size = num_kv_heads * head_dim             # 1024
qkv_size = q_size + 2 * kv_size              # 4096
gate_up_size = 2 * intermediate_size          # 6144 / 12288
q_heads_per_kv_head = num_attention_heads // num_kv_heads  # 2
```

**工厂方法：** `from_hf_config(hf_config, ...)` 从 HuggingFace config.json 构造。

#### 4.3.2 `variants.py` — 已知模型规格

```python
QWEN3_0_6B = Qwen3VariantSpec(
    canonical_name="qwen3-0.6b",
    aliases=("qwen3-0.6b", "qwen/qwen3-0.6b", "0.6b", ...),
    num_layers=28, hidden_size=1024, intermediate_size=3072,
    num_attention_heads=16, num_kv_heads=8, head_dim=128,
    vocab_size=151936, tie_word_embeddings=True,
)

QWEN3_1_7B = Qwen3VariantSpec(
    canonical_name="qwen3-1.7b",
    aliases=("qwen3-1.7b", "qwen/qwen3-1.7b", "1.7b", ...),
    num_layers=28, hidden_size=2048, intermediate_size=6144,
    num_attention_heads=16, num_kv_heads=8, head_dim=128,
    vocab_size=151936, tie_word_embeddings=True,
)
```

**识别函数：** `identify_qwen3_variant(config)` 通过 num_layers/hidden_size/intermediate_size/heads/kv_heads/head_dim/vocab_size 匹配。

**模型目录解析优先级：** `GR_QWEN3_1_7B_MODEL_DIR` → `QWEN3_1_7B_MODEL_DIR` → `GR_QWEN3_MODEL_DIR` → 参数默认值。

#### 4.3.3 `weights.py` — HF 权重映射

**核心变换：**

```
HF 权重名                              →  逻辑权重名
model.layers.{i}.self_attn.q_proj       ─┐
model.layers.{i}.self_attn.k_proj        ├→ layers.{i}.self_attn.qkv_proj  (concat dim=0)
model.layers.{i}.self_attn.v_proj       ─┘
model.layers.{i}.mlp.gate_proj          ─┐
model.layers.{i}.mlp.up_proj             ├→ layers.{i}.mlp.gate_up_proj    (concat dim=0)
                                          ─┘
model.layers.{i}.self_attn.q_norm       →  layers.{i}.self_attn.q_norm     (可选)
model.layers.{i}.self_attn.k_norm       →  layers.{i}.self_attn.k_norm     (可选)
model.embed_tokens.weight               →  embed_tokens.weight
model.norm.weight                       →  final_norm.weight
lm_head.weight                          →  不加载（tie_word_embeddings 时与 embed 共享）
```

**加载流程：**

```python
materialize_qwen3_checkpoint(model_dir, pack_qkv=True, pack_gate_up=True):
    loader = HFCheckpointLoader(model_dir)
    manifest = loader.manifest()                    # 发现 config + weight files
    adapter = Qwen3HFAdapter.from_manifest(manifest)
    plan = adapter.load_plan(pack_qkv=True, pack_gate_up=True)
    plan.validate(manifest)
    return plan.materialize(lambda name: loader.load_tensor(manifest, name))
```

#### 4.3.4 `loader.py` — Checkpoint 发现（模型无关）

```python
class HFCheckpointLoader:
    manifest() -> CheckpointManifest
        # 1. 读 config.json
        # 2. 查 safetensors index / pytorch index
        # 3. 有 index: weight_map → tensor_map + weight_files
        # 4. 无 index: glob *.safetensors / *.bin / *.pt

    load_tensor(manifest, name) -> Any
        # safetensors: safetensors.torch.load_file
        # pytorch: torch.load(path, map_location="cpu")
```

#### 4.3.5 `layers.py` — 层操作（1837 行核心实现）

**`Qwen3RMSNorm(nn.Module)` dispatch 链：**

```
flashinfer rmsnorm (CUDA) → F.rms_norm (PyTorch ≥2.4) → 手动实现 (float32 upcast)
```

**`TorchQwen3LayerOps` — 7 层 dispatch 链：**

```
┌─ input_norm:      flashinfer rmsnorm → F.rms_norm → 手动实现
│
├─ qkv:             TRT-LLM packed GEMM → torch.nn.Linear
│                   输出 split → [q(B,S,Hq,D), k(B,S,Hkv,D), v(B,S,Hkv,D)]
│
├─ qk_norm_rope:    TRT-LLM 融合(fused_qk_norm_rope, 在原始 qkv buffer 上原地操作)
│                   → SGLang inplace 融合(qknorm + rope)
│                   → 分离: q_norm → k_norm → rope
│
├─ rope_only:       SGLang inplace RoPE → flashinfer RoPE → 手动 cos/sin
│
├─ o_proj:          TRT-LLM packed GEMM → torch.nn.Linear
│
├─ post_attn_norm:  flashinfer fused_add_rmsnorm → 手动 add + norm
│                   (融合 residual + projected + RMSNorm 为一个 kernel)
│
└─ mlp:             SGLang silu_and_mul → torch.compile → TRT-LLM packed SiLU*Mul
                   → FusedMLP(torch) fallback
```

**`Qwen3SingleLayerPrefill` 三种前向模式：**

**`forward_prefill`：**

```
residual = hidden_states
normed = ops.input_norm(hidden_states)        # 可从上层 fusion 跳过
q, k, v = ops.qkv(normed)
q, k = ops.qk_norm_rope(q, k)
→ prefill attention + write ContextKV
residual, hidden_states = ops.post_attention_residual_norm(residual, attn_out)
mlp_out = ops.mlp(hidden_states)
→ 可选: next-layer norm fusion (fused_add_rmsnorm on mlp_out + residual)
return hidden_states, next_normed
```

**`forward_prefill_extend`（prefix cache 扩展）：**

```
position_ids = arange(prefix_len, prefix_len + suffix_len)  # 偏移位置
→ qkv → qk_norm_rope(position_ids=offset_ids)
→ write KV suffix into context_kv[..., prefix_len:prefix_len+suffix_len]
→ attention with full KV (prefix + suffix)
→ mlp
```

**`forward_decode`：**

```
residual = hidden_states
normed = ops.input_norm(hidden_states)
q, k, v = ops.qkv(normed)
position_ids = context_len + step              # decode 位置偏移
q, k = ops.qk_norm_rope(q, k, position_ids=position_ids)
→ BeamKVWriter.write_layer_step(layer_idx, step, width, k, v)
→ decode_engine.decode_attention_step(request, q, layer_idx, step, ...)
→ post_attention_residual_norm
→ mlp
→ 可选: next-layer norm fusion (gr_trtllm exact_fused_add_rmsnorm 或 flashinfer)
return hidden_states, next_normed
```

**Next-layer norm fusion（关键优化）：**

- 层 i 的 MLP 输出后，用 `fused_add_rmsnorm` 同时完成 `residual + mlp_out` 和 `RMSNorm`
- 结果作为层 i+1 的 `normed_hidden_states`，**跳过下一层的 input_norm 调用**
- 减少 N-1 次 RMSNorm kernel launch

#### 4.3.6 `model.py` — `Qwen3GRModel`

```python
class Qwen3GRModel(nn.Module):
    embed_tokens: nn.Embedding(vocab_size, hidden_size)
    layers: ModuleList[Qwen3SingleLayerPrefill] × num_layers
    norm: Qwen3RMSNorm(hidden_size)
    lm_head: nn.Linear(hidden_size, vocab_size)  # 或与 embed 共享
```

**5 个前向方法：**

| 方法 | 流程 |
|---|---|
| `forward_prefill` | embed → layer loop → norm → lm_head（可选 `last_token_logits_only`） |
| `forward_prefill_extend` | copy prefix KV → embed suffix → layer loop（offset positions）→ output |
| `forward_prefill_layer` | 单层 prefill（给 CUDA graph piecewise 用） |
| `forward_decode_step` | embed beam tokens → layer loop（每层写 BeamKV + decode attention）→ lm_head |
| `generate_fixed_beam` | 创建 `FixedBeamDecodeLoop` 并执行完整 beam search |

**`last_token_logits_only` 优化：**

Prefill 时只取 `hidden_states[:, -1, :]` 再过 lm_head，GEMM 从 `[B,S,H]@[V,H]` 降为 `[B,H]@[V,H]`。

**Flatten-linear 优化：**

对于 >2D 输入（如 decode 的 `[B, W, H]`），先 flatten 到 `[B*W, H]` 再 `F.linear`，避免低效的 batched matmul。可通过 `GR_INFERENCE_DISABLE_FLATTEN_LINEAR=1` 关闭。

**RoPE 缓存：**

- `_ROPE_COS_SIN_CACHE`：按 `(device, head_dim, theta, type, position/seq_len)` 缓存
- `_ROPE_POSITION_IDS_CACHE`：按 `(device, batch, seq_len, position/int)` 缓存
- `_SGLANG_ROPE_COS_SIN_CACHE`：SGLang 风格缓存

---

### 4.4 `gr_runtime/` — Beam Search 运行时

#### 4.4.1 Beam Search 核心（`beam_search.py`）

**`select_initial_topk`（Step 0）：**

```
输入: logits [1, S, V] 或 [1, V]
1. 取最后位置: logits[:, -1, :] → [1, V]
2. item_mask: masked_fill(~mask, -inf)
3. log_softmax (如果 score_mode="logprob")
4. topk(k=beam_width) → values[beam_width], indices[beam_width]
5. parent_beams = (0, 0, ..., 0)  # 全部来自唯一 root beam
6. → BeamSelection(token_ids, scores, parent_beams)
```

**`select_next_topk`（Step 1+）：**

```
输入: logits [1, W_prev, V]
1. candidate_logits = logits[0]                     # [W_prev, V]
2. item_mask: expand [V] → [W_prev, V], masked_fill -inf
3. log_softmax (如果 logprob)
4. scores = candidate_logits + previous_scores[:, None]   # 累积分数 [W_prev, V]
5. flat = scores.reshape(-1)                         # [W_prev * V]
6. topk(flat, k=beam_width)
7. 解码 flat indices:
     parent_beam[i] = flat_index[i] // vocab_size
     token_id[i]    = flat_index[i] %  vocab_size
8. → BeamSelection
```

**关键洞察：** 累积分数 `previous_score + current_logit` 实现标准 beam search score 累积。flat index 空间 `[W_prev * V]` 的编码为 `(beam, token)` 对。

#### 4.4.2 Batched Beam Search（`batched_beam_search.py`）

**`select_initial_topk_batched`：**

```
与单请求相同，但 topk 在 [B, V] 的 dim=-1 上批量执行
→ BatchedBeamSelection (同时持有 tensor 和 tuple 形式)
```

**`select_next_topk_batched` — 两阶段 topk 优化：**

```
输入: logits [B, W_prev, V]

Stage 1 (local topk):
    local_k = min(beam_width, vocab_size)
    topk(candidate_logits, k=local_k, dim=-1) → [B, W_prev, local_k]

Stage 2 (global topk):
    local_values += prev[:, :, None]           # [B, W_prev, local_k]
    flat = reshape [B, W_prev * local_k]
    topk(flat, k=beam_width, dim=-1)           # [B, beam_width]
    parent_beams = flat_indices // local_k
    token_ids = local_token_ids.gather(flat_indices)
```

**性能收益：** 当 V=151936, beam_width=256，复杂度从 `O(B * W * 151936)` 降为 `O(B * W * 256)`。

**`BatchedBeamSelection`：**

```python
@dataclass(frozen=True)
class BatchedBeamSelection:
    token_ids: tuple[tuple[int, ...], ...]          # [B][beam_width]
    scores: tuple[tuple[float, ...], ...]
    parent_beams: tuple[tuple[int, ...], ...]
    token_ids_tensor: Any | None                    # [B, beam_width]
    scores_tensor: Any | None                       # [B, beam_width]
    parent_beams_tensor: Any | None                 # [B, beam_width]
    materialize() -> BatchedBeamSelection           # 从 tensor 填充 tuple
```

#### 4.4.3 `batched_topk_indices.py` — Kernel 索引构建

```python
make_batched_topk_indices(path, Hq, decode_nums, beam_width)
    → tensor [B, 1, Hq, decode_nums, W]
```

**语义：** 对于每个当前 beam j，通过 `_beam_ancestry` 回溯其祖先在每步的 slot 编号，生成 `decode_idx * beam_width + ancestor_slot`。这告诉 decode attention kernel 从 BeamKV 的展平视图中 gather 哪些历史 KV。

**算法：**

```
if decode_nums == 1:
    identity mapping: beam j → slot j
else:
    for each batch, beam:
        ancestry = walk backward through beam_path entries
        index[decode_idx, beam] = decode_idx * beam_width + ancestor_slot
```

#### 4.4.4 `beam_kv_compaction.py` — BeamKV 历史压缩

**触发条件：** 动态 beam 缩窄后，某个 beam 的祖先 slot 超出 `active_beam_width`。

```
compact 前: slot 3 的祖先是 slot 7 (超出 active_width=4)
compact 后: slot 7 的 KV 复制到 slot 3
           → 索引变为 identity mapping
           → 可用 make_compacted_batched_topk_indices
```

**算法：**

```
for batch_idx, beam_path:
    for query_beam in [0, active_beam_width):
        ancestry = _beam_ancestry(beam_path, query_beam, decode_nums)
        for step in [0, history_steps):
            ancestor = ancestry[step]
            compact_kv[:, batch_idx, step, query_beam] = beam_kv[:, batch_idx, step, ancestor]
```

#### 4.4.5 `item_constraints.py` — Trie 约束解码

**`TokenTrie`：**

```python
class TokenTrieNode:
    children: dict[int, TokenTrieNode]
    terminal: bool = False
    item_ids: list[Any]     # 在此节点终止的 item

class TokenTrie:
    insert(sequence, item_id):
        node = root
        for token in sequence:
            node = node.children.setdefault(token, TokenTrieNode())
        node.terminal = True
        node.item_ids.append(item_id)

    allowed_next(prefix):
        node = traverse(prefix)
        return set(node.children.keys())
```

**`SemanticItem` + `SemanticItemCatalog`：**

```python
@dataclass(frozen=True)
class SemanticItem:
    item_id: Any
    token_ids: tuple[int, ...]      # item 的 token 序列
    metadata: Mapping[str, Any]

SemanticItemCatalog.from_records(records, item_id_field, token_ids_field, ...)
SemanticItemCatalog.from_jsonl(path, ...)
```

**`TrieItemMaskProvider` 掩码生成：**

```
initial_mask():
    allowed = trie.allowed_next(empty_prefix)
    → mask[V] boolean, True for allowed tokens

step_mask(logits[B, W, V]):
    for beam in [0, W):
        prefix = beam_path.token_trace(beam)    # 完整 token 历史
        allowed = trie.allowed_next(prefix)
        mask[beam, allowed] = True
    → mask[W, V]
```

**EOS 处理：**

```
if EOS already in prefix:
    allowed = {eos_token_id} only
elif allow_eos_for_terminal and trie.is_terminal(prefix):
    allowed.add(eos_token_id)
```

**`TrieItemMaskProviderStore` — 热替换协议：**

```python
snapshot()             # 获取当前 provider (加锁)
swap(provider, meta)   # 安装新 provider，bump version
reload_jsonl(path)     # 解析 JSONL → catalog → provider → swap
rollback()             # 恢复 previous provider
```

线程安全（`RLock`），支持 atomic reload/rollback + 版本追踪。

#### 4.4.6 `logits_processor.py` — Logits 处理管线

```python
@dataclass(frozen=True)
class TokenSuppressLogitsProcessor:
    token_ids: tuple[int, ...]
    fill_value: float = -inf
    phases: ("prefill", "decode")
    process_logits(logits, context):
        clone → logits[..., list(token_ids)] = -inf → return

@dataclass(frozen=True)
class TokenBiasLogitsProcessor:
    token_bias: tuple[tuple[int, float], ...]
    process_logits(logits, context):
        clone → logits[..., token_id] += bias → return
```

**管线执行：**

```
processed = logits
for processor in processors:
    processed = processor.process_logits(processed, context)
```

**工厂：** `logits_processor_from_spec` 从请求规范解析：`"token_suppress"` / `"suppress_tokens"` → TokenSuppressLogitsProcessor。

#### 4.4.7 Beam Width Policy（`beam_policy.py`）

| Policy | 行为 |
|---|---|
| `FixedBeamPolicy` | 始终返回 max beam width |
| `ScheduledBeamPolicy` | `{0: 256, 1: 128, 2: 64}` — 按 step 索引查表 |
| `ScoreMarginBeamPolicy` | 根据 score 集中度动态缩窄，`observe_scores()` 钩子 |

**beam width 确定（三级优先）：**

```
1. BeamWidthPolicy.width_for_step(step)          # 策略
2. item_mask_limited_beam_width(width, mask)      # 合法候选数限制
3. Hard cap: fixed_beam_width                     # 最大上限
```

#### 4.4.8 `decode_loop.py` — 固定宽度 Beam Decode Loop

```python
@dataclass(frozen=True) DecodeLoopStep:
    step, logits, token_ids, parent_beams, scores

@dataclass(frozen=True) DecodeLoopResult:
    generation: GRGenerationState
    steps: tuple[DecodeLoopStep, ...]
    stop_reason: str
    final_token_ids: tuple[int, ...]   # property

@dataclass FixedBeamDecodeLoop:
    model, decode_engine
    item_masks, item_mask_provider
    beam_width_policy
    stop_token_ids, logits_processors
```

**主循环算法：**

```
1. 初始化:
   a. 获取 initial_item_mask
   b. 应用 logits processors (phase="prefill", step=0)
   c. beam_width = policy.width_for_step(0)
   d. select_initial_topk → beam_path.append

2. Decode Loop (step 0..max_steps-1):
   a. logits = model.forward_decode_step(beam_token_ids, generation, engine, step, width)
   b. 应用 logits processors (phase="decode")
   c. item_mask = trie step_mask 或预设
   d. next_width = policy.width_for_step(step+1), clamp by item_mask
   e. select_next_topk(logits, prev_scores, next_width, mask)
   f. 记录 DecodeLoopStep
   g. 检查停止条件:
      - 所有 beam 命中 stop_token → "stop_token"
      - 所有 beam 在 trie 上终止 → "item_complete"
   h. beam_token_ids = tensor(selection.token_ids)

3. return DecodeLoopResult(generation, steps, stop_reason || "max_decode_steps")
```

#### 4.4.9 `engine.py` — GRDecodeEngine

```python
@dataclass
class GRDecodeEngine:
    attention: GRDecodeAttention
    fixed_beam_width: int

    decode_attention_step(request, q, *, layer_idx, step, ...):
        validate request state
        → build GRDecodeAttentionInputs
        → self.attention(inputs)
        → DecodeStepOutput
```

#### 4.4.10 `request.py` — GRRequestState

```python
@dataclass
class GRRequestState:
    request_id: str
    context_kv: ContextKV
    beam_kv: BeamKV
    beam_path: BeamPath

    validate():
        context_kv.num_layers == beam_kv.num_layers
        context_kv.batch_size == beam_kv.batch_size
        beam_path.max_decode_steps ∈ {beam_kv.max_decode_steps, +1}
        beam_path.max_beam_width == beam_kv.max_beam_width
```

#### 4.4.11 `timing.py` — 时序记录

```python
TimingRecorder.section(name):    # context manager
    cuda.synchronize()
    start = perf_counter()
    yield
    cuda.synchronize()
    totals_ms[name] += (perf_counter() - start) * 1000
```

可选 NVTX range (`GR_INFERENCE_NVTX=1`)。

#### 4.4.12 张量形状汇总

| 阶段 | 张量 | 形状 |
|---|---|---|
| Prefill logits | logits | `[1, S, V]` 或 `[1, V]` |
| Initial topk | scores | `[V]` |
| Decode logits | logits | `[1, W_prev, V]` |
| Next topk (cumulative) | flat scores | `[W_prev * V]` |
| item_mask (initial) | mask | `[V]` |
| item_mask (decode, single) | mask | `[W_prev, V]` |
| item_mask (decode, batched) | mask | `[B, W_prev, V]` |
| BeamKV key/value | kv | `[L, B, max_steps, max_W, Hkv, D]` |
| ContextKV key/value | kv | `[L, B, context_len, Hkv, D]` |
| topk_indices (kernel) | indices | `[B, 1, Hq, decode_nums, W]` |
| beam_token_ids | tokens | `[B, W]` |
| BatchedBeamSelection tensors | all | `[B, beam_width]` |

---

### 4.5 `gr_serving/` — 连续批调度与 HTTP Serving（~8900 行）

#### 4.5.1 连续批调度状态机（`continuous.py`，~3560 行）

**状态流转：**

```
                    submit()
                       │
                       ▼
              ┌─ waiting_prefill ──┐
              │                    │ _admit_prefill_batch()
              │                    │ (分配 KV lease)
              │                    ▼
              │              ┌─ decoding ─────────┐
              │              │  step++             │
              │              │  forward_decode     │
              │              │  beam selection     │
              │              │  stop check ────────┤
              │              │                     │
              │              │  max_steps reached  │
              │              │  or stop_token      │
              │              │  or item_complete   │
              │              └─────────┬───────────┘
              │                        │
              │                        ▼
              └────────────────── finished
                                (释放 KV lease)
```

**`GRContinuousScheduler.tick()` 四步：**

1. `_fail_timed_out_requests()` — 超时检测
2. `_admit_prefill_batch()` — 从 waiting_prefill 取 ≤ `max_prefill_batch_size` 个请求，分配 KV lease
3. `_plan_decode_batches()` — 按 `(step, beam_width, next_beam_width, context_len)` 分组
4. 执行 decode batches

**`GRContinuousRequestState` 字段：**

```python
stage: "waiting_prefill" | "decoding" | "finished"
request, current_decode_step
submitted_tick, admitted_tick, finished_tick
generation: GRGenerationState | None
kv_lease, beam_kv_pool_lease, context_kv_pool_lease
active_beam_width
# Tensor 快速路径追踪:
decode_selection_token_ids, decode_selection_scores, decode_parent_history
pending_decode_*
```

#### 4.5.2 `GRContinuousServingExecutor` — 执行引擎

**Prefill 执行路径：**

```
_run_prefill_requests():
    按 input_ids shape 分组
    → 检查 prefix cache (exact/prefix match)
    → cache miss:  concat input_ids → _forward_prefill (CUDA graph / eager)
    → prefix hit:  forward_prefill_extend 或 decode-step-by-step 扩展
    → exact hit:   复制 KV 到 pool slot
    → 切片 per-request PrefillResult
    → initial beam selection
    → 分配 beam_kv from pool
    → 初始化 GRGenerationState + tensor-backed decode state
```

**Decode 执行路径（标准）：**

```
_run_decode_batch():
    构建 batched generation (concat 或 alias pool views)
    → 构建 topk_indices
    → model.forward_decode_step
    → logits processors + item mask
    → select_next_topk_batched
    → scatter beam KV 回 per-request slices
    → 检查 stop reasons per request
```

**Decode 执行路径（Tensor 快速路径）：**

```
条件: 无 item_mask, 无 logits_processors, 无 stop_tokens, 稳定 beam_width, CUDA tensors

_run_decode_batch_tensor_selection():
    GPU-resident tensors (decode_selection_token_ids/scores/parent_history)
    → beam selection materialize=False (纯 GPU，不转 CPU)
    → 累积到 pending_decode_* 列表
    → 结束时 _flush 重建 BeamPath entries
```

**优雅降级：** batched decode 失败时自动降级为 sequential per-request 执行。

#### 4.5.3 三层内存系统

**Tier 1: `GRKVLeaseAllocator`（元数据容量跟踪）**

```python
allocate(request_id, context_tokens, beam_slots) -> GRKVLease
release(request_id) -> GRKVLease | None
can_allocate(...) -> bool   # max_running_requests, max_context_tokens, max_beam_slots
leases: dict[request_id, GRKVLease]
```

**Tier 2: `GRPagedKVLeaseAllocator`（页式扩展）**

```python
# 继承 Tier 1，增加页式管理
context_page_size, beam_page_size
free_context_pages: list[int]
free_beam_pages: list[int]
# 跟踪: 内部碎片, max_used_pages (高水位)
```

**Tier 3: Dense GPU Tensor Pools**

```python
GRDenseContextKVPool:
    shape: [layers, slots, max_context_len, kv_heads, head_dim]
    allocate(request_id, context_len) → slice [:, slot:slot+1, :ctx_len]
    allocate_batch(request_ids, context_len) → _first_free_run 找连续 slot
    release(request_id)

GRDenseBeamKVPool:
    shape: [layers, slots, max_decode_steps, max_beam_width, kv_heads, head_dim]
    allocate(request_id) → slice [:, slot:slot+1]
    release(request_id)
```

**连续 slot 优化：** 分配连续 slot 时直接通过 tensor view alias（跳过 `torch.cat`）。

**内存估算：**

```
kv_unit = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
context_kv_bytes = batch * context_len * kv_unit
beam_kv_bytes = batch * max_steps * max_width * kv_unit
logits_workspace = batch * active_width * vocab_size * 4  (FP32)
```

#### 4.5.4 CUDA Graph 管理

**Prefill CUDA Graph（`GRPrefillCudaGraphRunner`）：**

```
模式: "piecewise" (默认) 或 "full"
piecewise: embed → 4 个 layer chunk → output = 6 个 graph piece
Key = (device, dtype, input_shape, ctx_kv_key_shape, ctx_kv_val_shape,
       ctx_kv_key_view, ctx_kv_val_view, last_token_logits_only)
Replay 校验: context_kv key/value 的 data_ptr() + storage_offset() 必须匹配
```

**Decode CUDA Graph（`GRDecodeCudaGraphRunner`）：**

```
Key = (device, dtype, beam_token_ids_shape, context_len,
       ctx_kv_key_shape, beam_kv_key_shape,
       4 个 KV view_key (data_ptr + storage_offset),
       topk_shape, step, width, decode_nums)

Replay 校验: 检查 4 个 KV tensor 的 data_ptr() + storage_offset()
  → 全部匹配: copy inputs → replay graph
  → 任一不匹配: 重新 capture (除非 frozen)

LRU 驱逐, max_entries=128 (GR_INFERENCE_DECODE_CUDA_GRAPH_MAX_ENTRIES)
Side stream warmup

启动 warmup: 覆盖常见 batch/pool window shapes
warmup 后冻结: GR_FREEZE_CUDA_GRAPHS_AFTER_WARMUP=1
```

**Bucket padding：**

```
GR_INFERENCE_DECODE_CUDA_GRAPH_BATCH_BUCKETS=1,2,4,8
实际 batch=3 → pad 到 4，复用已捕获的 graph
需要 context_kv 和 beam_kv 在 pool 中连续
replay 后 output 切片回实际 batch size
```

#### 4.5.5 Prefix Cache（`prefix_cache.py`）

```python
class GRPromptPrefixCache:
    _entries: OrderedDict[(extra_key, tokens), _GRPrefixEntry]   # LRU
    _radix_tree: _GRPrefixNode                                    # 前缀匹配

    insert(input_ids, prefill, extra_key):
        page-aligned tokens → entries → rebuild radix tree
        LRU eviction if over max_entries/max_tokens

    match(input_ids, extra_key):
        exact match → prefix match (deepest common prefix via radix tree)
```

**集成：**

```
exact hit:   复制 KV 到 pool slot（零计算）
prefix hit:  forward_prefill_extend 扩展 KV cache
cache miss:  完整 prefill → 存入 cache
```

#### 4.5.6 HTTP API（`http.py`）

**路由表：**

| Method | Path | 说明 |
|---|---|---|
| GET | `/health`, `/ready` | 健康检查 |
| GET | `/config`, `/build` | 配置和构建信息 |
| GET | `/status`, `/metrics` | 状态和指标 |
| GET | `/metrics/prometheus` | Prometheus 格式 |
| POST | `/generate` | **SGLang 兼容的阻塞式 beam search 生成** |
| POST | `/submit` | 异步提交（202） |
| POST | `/submit_many` | 批量提交（max 32） |
| GET | `/poll/{id}`, `/result/{id}` | 异步结果查询 |
| POST | `/catalog/reload` | Item 目录热替换 |
| POST | `/catalog/rollback` | Item 目录回滚 |
| POST | `/cancel`, `/drain`, `/shutdown` | 生命周期管理 |

**`/generate` SGLang 兼容性：**

```
sampling_params.max_new_tokens → max_decode_steps = max(1, max_new_tokens - 1)
sampling_params.n → beam_width
阻塞轮询，超时 300s
返回 SGLang-equivalent beam_results
```

**认证：** 可选 `X-GR-API-Key` header 或 Bearer token。`/health` 和 `/ready` 免认证。

#### 4.5.7 指标体系

```
Scheduler:
  waiting_prefill_requests, decoding_requests, finished_requests
  submitted/succeeded/cancelled/failed/timed_out
  avg_prefill_batch_size, avg_decode_batch_size
  kv_allocator: running/context_tokens/beam_slots/utilization
  kv_health: leak_detected, orphaned/missing leases

Executor:
  prefill_ms, decode_ms, total_ms
  prefix_cache: hits/misses/exact/prefix/extend_tokens
  decode_inputs_cache, topk_indices_cache: hits/misses
  decode_cuda_graph: captures/replays/hits/misses/dynamic_skips
  beam_kv_pool/context_kv_pool: usage/utilization/high_water_mark

Worker:
  worker_ticks, worker_errors, batch_fill_waits

HTTP /metrics/prometheus: 所有数值指标加 gr_serving_ 前缀
KV event log: allocate/release 事件 (bounded=128)
```

#### 4.5.8 请求生命周期

```
1. Submit:
   HTTP POST /generate → GRHTTPServingAdapter.handle()
   → GRServingWorker.submit() → facade.submit_many() → executor.submit()
   → scheduler.submit() → stage="waiting_prefill"

2. Prefill admission:
   scheduler.tick() → _admit_prefill_batch()
   → allocate KV lease → stage="decoding"
   → executor callback: prefill execution

3. Prefill execution (executor):
   prefix cache check → CUDA graph or eager prefill
   → beam_kv pool allocate
   → initial beam selection → init GRGenerationState

4. Decode planning:
   _plan_decode_batches() → group by (step, width, ctx_len)

5. Decode execution (per batch):
   batched generation (pool view alias)
   → CUDA graph replay or eager decode
   → beam selection (tensor fast path or standard)
   → scatter KV → stop check

6. Finish:
   stage="finished" → release KV lease
   → release beam_kv/context_kv pool leases
   → store GRServingResponse

7. HTTP response:
   beam_results (SGLang-equivalent output)
```

---

## 五、端到端交互全流程

### 离线推理路径

```
Qwen3HFAdapter.load_plan()
  → HFCheckpointLoader.manifest()
  → plan.materialize()
  → Qwen3GRModel.load_logical_weights()

Qwen3GRModel.forward_prefill(input_ids, context_kv)
  → embed → 28 层 forward_prefill → norm → lm_head
  → 每层: input_norm → qkv → qk_norm_rope → prefill attention → write ContextKV
         → post_attn_norm → mlp (可选 next-layer norm fusion)
  → PrefillResult(logits, context_kv)

GRGenerationState.from_prefill(prefill_result, ...)
  → allocate BeamKV [L, B, max_steps, max_width, Hkv, D]
  → BeamPath(max_steps+1, max_width)

FixedBeamDecodeLoop.run(generation, max_steps)
  → select_initial_topk → beam_path.append
  → for step: forward_decode_step → select_next_topk → append
  → 每层: input_norm → qkv → qk_norm_rope → write BeamKV
         → GRDecodeEngine.decode_attention_step(ContextKV + BeamKV + BeamPath + topk_indices)
         → post_attn_norm → mlp (可选 next-layer norm fusion)
  → DecodeLoopResult
```

### 在线 Serving 路径

```
HTTP POST /generate
  → GRHTTPServingAdapter.handle()
  → GRServingWorker.submit()
  → GRContinuousScheduler.submit() → stage="waiting_prefill"

tick():
  → _admit_prefill_batch() → allocate KV lease
  → executor._run_prefill_requests():
      → prefix cache check
      → CUDA graph or eager prefill
      → beam_kv pool allocate
      → initial beam selection
  → stage="decoding"

tick():
  → _plan_decode_batches() → group by (step, width, ctx_len)
  → executor._run_decode_batch():
      → batched generation (pool view alias)
      → topk_indices construction
      → CUDA graph replay or eager decode
      → beam selection (tensor fast path or standard)
      → scatter KV back
      → stop check
  → stage="finished" → release KV lease

HTTP response → beam_results (SGLang-equivalent output)
```

---

## 六、关键设计决策总结

| 决策 | 理由 |
|---|---|
| **ContextKV 密集连续** | kernel 友好 + CUDA graph 所需的指针稳定性 |
| **BeamKV step-major 布局** | 每个 step 的 beam 连续写入，无需跨步访问 |
| **BeamPath append-only log** | O(1) 追加，O(T) 回溯，T 通常 ≤ 5 |
| **两阶段 topk** | `O(B*W*beam_width)` vs `O(B*W*V)`，V=151936 |
| **Tensor 快速路径** | beam selection 留在 GPU，避免 CPU round-trip |
| **Pool view alias** | 连续 slot 时跳过 `torch.cat` |
| **Bucket padding** | 实际 batch pad 到固定 bucket 复用 CUDA graph |
| **Next-layer norm fusion** | 减少 N-1 个 RMSNorm kernel launch |
| **JIT CUDA kernel 按需启用** | 大部分默认 off，仅 fused_qk_norm_rope 和 exact_add_rmsnorm 默认 on |
| **三段式内存** | 元数据容量 → 页式扩展 → Dense GPU Pool，各层职责清晰 |
| **优雅降级** | batched decode 失败 → sequential；CUDA graph 失败 → eager |
| **能力导向 kernel registry** | 一个能力可配多个 backend，环境变量覆盖，profile 持久化 |
| **SGLang 兼容 `/generate`** | 可直接用 SGLang 的 `bench_serving` 客户端做公平对比 |
