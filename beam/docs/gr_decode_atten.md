# [Beam Search Decode Attention](https://github.com/NVIDIA/recsys-examples/blob/a4d3b013071c3709068d69ccf4caafcfdaffcba7/corelib/gr_decode_atten/README.md?plain=1#L1) — 设计与实现文档

## 1. 项目动机

### 1.1 问题背景

SID-GR (Semantic ID Generation & Retrieval) 模型在推理阶段使用 beam search 逐层生成 semantic ID token。每次 beam step 需要对**全部历史序列**执行注意力计算。

原始的 `generate()` 方法不使用 KV cache——每步 decode 都重新对完整历史做 attention。随着用户交互历史增长（256 → 512 → 1024+ tokens），decode 阶段延迟线性增长，成为系统瓶颈。

### 1.2 为什么标准 KV cache 不够

标准 LLM 解码的 KV cache 优化假设 token 是顺序追加的。Beam search 场景下，beam 之间会 merge/split（通过 topK 选择），beam KV 不是简单追加而是需要 **gather** 操作：

- 每步 beam search 可能选择不同的 topK 路径
- Beam KV 的访问模式是不规则的（topK gather），而非连续的
- Context KV（历史 prompt）所有 beam 共享，访问模式是 dense 的

这需要一个专门为 beam search 设计的 decode attention kernel，将 KV cache 拆分为 context/beam 两部分并分别高效处理。

### 1.3 目标

设计并实现一个高性能 beam search decode attention kernel，使得 decode 阶段不再需要重新 attend 全部历史，从而在长序列场景下获得数量级的加速。

---

## 2. 设计

### 2.1 KV Cache 拆分

将 KV cache 分为两部分，对应不同的访问模式：

| 属性 | Context KV | Beam KV |
|------|-----------|---------|
| 来源 | Prefill 阶段编码 | Decode 阶段逐步产生 |
| 共享性 | 所有 beam 共享 | 每 beam 独立 |
| 访问模式 | Dense，顺序扫描 | Sparse，topK gather |
| 典型长度 | 数千 tokens | `decode_nums` × `beam_width` tokens |

### 2.2 3-Kernel Pipeline

将 beam decode attention 数学上分解为三个独立 kernel，每个使用最优的硬件策略：

```
┌─────────────────────────────────────────────────────────┐
│  K1: Context Attention                                   │
│  Q × Context KV                                          │
│  硬件: Tensor Core MMA (SM80/90/100/120)                 │
│  特性: split-KV 并行, TMA/cp.async 数据加载               │
│  输出: fp32 partial (O_k1, LSE_k1)                       │
├─────────────────────────────────────────────────────────┤
│  K2: Beam Sparse Attention                               │
│  Q × Beam KV[topK]                                       │
│  硬件: CUDA Core scalar FMA                               │
│  特性: topK gather 加载, online softmax                   │
│  输出: fp32 partial (O_k2, LSE_k2)                       │
├─────────────────────────────────────────────────────────┤
│  K3: Combine                                             │
│  Numerically stable log-sum-exp merge                    │
│  输入: K1 的 ns 个 partial + K2 的 1 个 partial           │
│  输出: bf16/fp16 O, fp32 LSE                             │
└─────────────────────────────────────────────────────────┘
```

**分解的优势**：
- K1 处理长序列 dense attention，使用 Tensor Core + split-KV 充分利用 SM
- K2 处理短序列 irregular gather，使用 CUDA Core 避免不规则访问的 Tensor Core 开销
- K3 合并多个 partial 结果，使用 numerically stable 的 log-sum-exp

### 2.3 Fused Kernel 路径

在 SM8x/SM90/SM120 上，将 K1+K2 融合为单次 kernel launch：

```
Fused (ns=1):  1 kernel launch  → context + beam fused → bf16 output
Fused (ns>1):  2 kernel launches → fused (ns fp32 partials) + K3 combine → bf16 output
3-kernel:      3 kernel launches → K1 (ns fp32 partials) + K2 (1 fp32 partial) + K3 combine → bf16 output
```

Beam sparse phase 在 context attention mainloop 之后，直接在 MMA accumulator 上执行 CUDA core FMA，共享 softmax 状态。减少 global memory roundtrip。

SM100 (B200) 上 fused beam path 性能不如 3-kernel，因此仅支持 3-kernel 路径。

### 2.4 GQA 支持

通过 `PackGQA` utility 支持Grouped Query Attention (GQA) 和 Multi-Query Attention (MQA)：

- 同一 KV head 组内的 Q head 共享 topK indices
- `topk_indices` 按 `qhead_per_kvhead` 步长采样，每个 KV head 组只 gather 一次 K/V
- 自动检测 MHA (`qhead_per_kvhead=1`) vs GQA

### 2.5 变长历史支持

提供两种机制处理变长 context 序列：

| 参数 | 模式 | 形状 | 说明 |
|------|------|------|------|
| `seqused_k` | Dense + valid-length | `[B]` int32 | K context 仍是 `[B, Sk, H, D]`，`seqused_k[b]` 之后的位置被 mask |
| `cu_seqlens_k` | Jagged | `[B+1]` int32 | K context 变为 `[total_k, H, D]`，无 padding 计算 |

两者互斥，仅 `backend="3kernel"` 支持。

---

## 3. 实现

### 3.1 项目结构（42 files, ~13,000 行）

```
gr_decode_atten/
├── interface.py                    # 唯一公共 API: beam_decode_attn()
├── src/
│   ├── common/                     # 共享基础设施 (5,030 行)
│   │   ├── kernel_config.py        #   KernelConfig dataclass
│   │   ├── pack_gqa.py             #   GQA layout 变换
│   │   ├── paged_kv.py             #   Paged KV 管理
│   │   ├── pipeline.py             #   多级流水线同步
│   │   ├── softmax.py              #   Online softmax
│   │   ├── tile_scheduler.py       #   Work tile 分配
│   │   ├── mask.py                 #   Causal / sliding window mask
│   │   ├── block_info.py           #   Block 索引计算
│   │   ├── seqlen_info.py          #   Variable-length 序列信息
│   │   ├── barrier.py              #   Named barrier 同步
│   │   ├── copy_utils.py           #   TMA / cp.async copy 工具
│   │   ├── cute_dsl_utils.py       #   CuTe DSL 辅助函数
│   │   ├── fast_math.py            #   exp2 / log2 快速近似
│   │   ├── fa_logging.py           #   调试日志
│   │   └── utils.py                #   通用工具
│   ├── sm80/flash_fwd.py           # K1: Ampere (A100/L40)     — 1,456 行
│   ├── sm90/flash_fwd.py           # K1: Hopper (H100/H800)    — 1,911 行
│   ├── sm100/flash_fwd.py          # K1: Blackwell (B200)      — 3,132 行
│   ├── sm120/flash_fwd.py          # K1: RTX PRO6000 (继承SM80) —   76 行
│   ├── decode/flash_fwd.py         # K2: Beam Sparse Attention  —   540 行
│   └── flash_fwd_combine.py        # K3: Combine                —   781 行
└── tests/                          # 测试与基准
    ├── reference.py                #   PyTorch golden reference
    ├── test_fwd.py                 #   E2E 测试 (672 cases)
    ├── test_context.py             #   K1 单元测试 (144 cases)
    ├── test_beam.py                #   K2 单元测试 (384 cases)
    ├── benchmark.py                #   性能基准 & ncu/nsys profiling
    └── conftest.py                 #   pytest 配置
```

### 3.2 实现语言与框架

所有 kernel 使用 **NVIDIA CuTe DSL**（CUTLASS 的 Python DSL）编写，通过 `cute.compile` JIT 编译为 CUDA kernel。

这不是伪代码或接口定义——每个文件都是可编译执行的完整 GPU kernel 实现。CuTe DSL 是 CUTLASS 生态的现代开发方式，用 Python 描述 kernel 语义（tensor layout、copy atom、MMA atom、pipeline 同步），编译器生成 PTX。

### 3.3 K1: Context Attention — 多架构实现

#### SM80 (A100/L40/L20) — `src/sm80/flash_fwd.py` (1,456 行)

- 数据加载: `cp.async` pipeline，2-stage 预取 K/V
- 计算: `mma.sync.aligned.m16n8k16` Tensor Core 指令
- SMEM: Q tile + (K tile + V tile) × num_stages
- 支持 fused beam: MMA accumulator 上直接执行 CUDA core FMA

#### SM90 (H100/H800/H20) — `src/sm90/flash_fwd.py` (1,911 行)

在 SM80 基础上增加：
- 数据加载: TMA (Tensor Memory Accelerator) 加载 Q/K/V
- 计算: HGMMA (Hopper GEMM), 4-thread parallel QK
- 支持 fused beam + split-KV

#### SM100 (B200/B300) — `src/sm100/flash_fwd.py` (3,132 行)

完全重写，是该库最复杂的实现：

```
Warp 布局 (16 warps):
  Warp  0-3:  softmax group 0
  Warp  4-7:  softmax group 1
  Warp  8-11: correction (rescale O, compute final LSE)
  Warp   12:  MMA (tcgen05)
  Warp   13:  epilogue (SMEM → gmem)
  Warp   14:  load (TMA)
  Warp   15:  CLC scheduler / empty

TMEM (Tensor Memory) 布局:
  S (QK scores): offset 0, 128
  P (softmax output): offset 64, 192
  O (accumulation): after S + P

关键特性:
  - tcgen05 MMA 指令
  - 2-CTA cluster MMA instructions
  - split_P_arrive: P column write 与 P@V MMA 重叠
  - CLC (Cooperative Launch Cluster) scheduler
  - TMEM 管理 (allocate / deallocate per tile)
  - SM103 exp2 原生指令 vs SM100 exp2 emulation
```

#### SM120 (RTX PRO6000) — `src/sm120/flash_fwd.py` (76 行)

继承 SM80 实现，仅覆写 SMEM 容量检查（99KB vs SM80 的 163KB）。使用 `arch=80` 确保 cp.async 代码路径。

### 3.4 K2: Beam Sparse Attention — `src/decode/flash_fwd.py` (540 行)

```python
线程布局 (FlashInfer pattern):
  tx (bdx) = head_dim / vec_size    # 一个 warp 覆盖完整 head_dim
  ty (bdy) = qhead_per_kvhead       # GQA 组内头并行
  tz (bdz) = num_threads / (bdx * bdy)  # KV tile 行并行

Grid: (batch × beam_width, kv_heads, 1)
```

执行流程:

```
Step 1: Q → 寄存器
  向量化 gmem → register, 无 SMEM 开销

Step 2: K/V 加载
  Sparse path: _gather_load_tile()
    按 topk_indices gather K/V 到 SMEM (同步加载)
    copy_threads_per_row 线程协作加载一行
  Dense path:
    cp.async 2-stage pipeline 预取 K/V

Step 3: Online Softmax (log2 域)
  QK dot product → warp_reduce
  row_max → exp2(scale - max) → row_sum
  累积: O = O × exp2(m_prev - m_new) + P × V

Step 4: Multi-warp Merge (bdz > 1)
  各 warp 的 softmax 状态 (m, d, O) 写入 SMEM
  warp 0 负责合并所有 warp 的结果

Step 5: Epilogue
  O = O / row_sum (使用 rcp_approx 快速倒数)
  转换为目标 dtype (bf16/fp16)
  LSE = (row_max × scale_log2 + log2(row_sum)) × LN2
```

关键设计：使用 **CUDA Core scalar FMA** (`MmaUniversalOp`) 而非 Tensor Core，因为 beam KV 的 gather 访问模式不规则，Tensor Core 的 tile 加载不高效。

### 3.5 K3: Combine — `src/flash_fwd_combine.py` (781 行)

移植自 FlashAttention 的 `flash_fwd_combine_kernel.h`，从 CUTLASS C++ 重写为 CuTe DSL。

```
执行流程:
  1. 加载所有 split 的 LSE_partial 到 SMEM (cp.async)
  2. 计算全局 max LSE → exp scale → sum → log → 最终 LSE
  3. 加载 O_partial 到 SMEM (4-stage pipeline 预取)
  4. 按权重累加: O = Σ scale[s] × O_partial[s]
  5. 写最终 O 和 LSE 到 gmem

优化:
  - LSE SMEM 使用 swizzle 消除 bank conflict
  - SM90/SM100 支持 PDL (Programmatic Dependent Launch)
  - FastDivmodDivisor 用于高效索引计算
  - max_valid_split 短路: 跳过无效 split 的 O_partial 加载
```

### 3.6 Split-KV 启发式

`num_splits_heuristic()` 决定 K1 的 KV 分割数，移植自 FlashAttention Hopper 的 `heuristics.h`：

- 目标: 最大化 SM 利用率 (≥85% 的最优效率)
- 大 workload (total_mblocks ≥ 0.8 × num_SMs): 不 split（除非 KV 超过 L2 cache）
- 小 workload: 找到满足 ≥85% 效率的最小 split 数
- 限制: 已知问题 — split-KV + `seqused_k` 在 SM90 上会 hang，自动 force ns=1

### 3.7 运行时加载与 Fallback

SID-GR 模型通过 `sys.path` 动态加载 kernel：

```python
# jagged_flash_attn_block.py
def _ensure_gr_decode_atten_on_path():
    gr_decode_dir = os.path.join(repo_root, "corelib", "gr_decode_atten")
    sys.path.insert(0, gr_decode_dir)

def _get_beam_decode_attn():
    _ensure_gr_decode_atten_on_path()
    from interface import beam_decode_attn
    return beam_decode_attn
```

加载失败时（无 CuTe DSL 环境），自动 fallback 到纯 PyTorch reference 实现：

```python
# _beam_decode_attn_reference(): 单精度数学等价的 PyTorch 实现
# 支持 dense / sparse / GQA / seqused_k 全部语义
```

---

## 4. 接口说明

### 4.1 公共 API: `beam_decode_attn()`

```python
from interface import beam_decode_attn

out, lse = beam_decode_attn(
    q,                          # [B, 1, W, Hq, D]  bf16/fp16
    k_context,                  # [B, Sk, Hkv, D]  或 jagged [total_k, Hkv, D]
    v_context,                  # 同 k_context
    k_beam,                     # [B, dn*W, Hkv, D]
    v_beam,                     # 同 k_beam
    topk_indices,               # [B, 1, Hq, max_dn, W]  int32
    decode_nums,                # 当前 decode 步数（含本步）
    softmax_scale=None,         # 默认 1/sqrt(dim)
    return_lse=False,           # 是否返回 log-sum-exp
    out=None,                   # 预分配输出 buffer
    backend="dsl",              # "dsl" (fused) 或 "3kernel"
    seqused_k=None,             # [B] int32, 变长 context (dense mode)
    cu_seqlens_k=None,          # [B+1] int32, jagged offsets
)
# 返回:
#   out: [B, 1, W, Hq, D]  同输入 dtype
#   lse: [B, 1, W, Hq]     fp32 (或 None)
```

### 4.2 参数约束

| 参数 | 约束 |
|------|------|
| `backend="dsl"` | 默认值，SM8x/SM90/SM120 使用 fused kernel。**不支持** `seqused_k` 和 `cu_seqlens_k` |
| `backend="3kernel"` | SM100 强制使用。**支持** `seqused_k` 和 `cu_seqlens_k` |
| `seqused_k` | 仅 `"3kernel"`，与 `cu_seqlens_k` 互斥 |
| `cu_seqlens_k` | 仅 `"3kernel"`，与 `seqused_k` 互斥。此时 k/v_context 为 3-D jagged |
| `k_beam.shape[1]` | 必须等于 `decode_nums × beam_width` |
| `head_q % head_kv` | 必须整除（GQA 支持） |
| `seqlen_q` | 必须为 1（decode 模式，每次只处理 1 个 query token） |

### 4.3 Backend 选择建议

| 场景 | 推荐 backend | 原因 |
|------|-------------|------|
| B200 (SM100) | `"3kernel"` (唯一选项) | fused beam 在 SM100 上更慢，代码自动降级 |
| H100 + uniform history | `"dsl"` | 单次 kernel 调用比 3-kernel 快 ~1.5x |
| H100 + 变长 history | `"3kernel"` | `"dsl"` 不支持 `seqused_k` |
| A100 | `"dsl"` | fused + split-KV |
| 无 CuTe DSL 环境 | 任意 | 自动 fallback 到 PyTorch reference |

### 4.4 内部 Kernel 接口

虽然不直接暴露给用户，但理解有助于调优：

| 函数 | 位置 | 功能 |
|------|------|------|
| `_context_attention()` | `interface.py:195` | K1: Q × Context KV，自动选 SM80/90/100/120 |
| `_beam_sparse_attention()` | `interface.py:335` | K2: Q × Beam KV[topK]，CUDA Core FMA |
| `_combine()` | `interface.py:423` | K3: log-sum-exp 合并 partials |
| `_fused_context_beam()` | `interface.py:514` | Fused K1+K2，仅 SM8x/SM90/SM120 |

### 4.5 在 SID-GR 模型中的调用链

```
gpt_model.generate_beam_decode()          # 用户入口
  ├── model.prefill(history_tokens)        # 编码历史，缓存 context KV
  │     └── JaggedGPTModel.prefill()       # 逐层缓存 (k, v)
  └── decode loop (per hierarchy step):
        ├── beam_search.propagate()        # topK 选择，产生 topk_indices
        └── model.decode_beam()            # 逐层 decode
              └── JaggedGPTLayer.decode_beam()
                    └── beam_decode_attn()  # 调用 CuTe kernel 或 PyTorch fallback
```

---

## 5. 效果

### 5.1 端到端性能（H100 PCIe, torch 2.11.0a0+nv26, CUDA 13）

#### 短序列模型（hidden=256, layers=2, hierarchies=3）

| 配置数 | 加速比范围 | 中位数 |
|--------|-----------|--------|
| 24 | 1.31x ~ 1.43x | **1.38x** |

加速比较为稳定，因为短序列下 context KV 计算量占比有限。

#### 长序列模型（hidden=512, layers=8, hierarchies=4）

| hist_len | 加速比 |
|----------|--------|
| 256 | **~2.1x** |
| 512 | **~4.7x** |
| 1024 | **~11-12x** |

加速比随历史长度**超线性增长**。原因：`generate()` 每步 attention 计算量与序列长度成正比 O(Sk)，而 `generate_beam_decode()` 的 context KV 已缓存，仅计算 O(decode_nums × W) 的 beam attention。

#### 阶段耗时分解（短序列模型）

| 阶段 | 耗时 |
|------|------|
| Prefill | ~3 ms |
| Decode loop | ~5.7 ms |

Prefill 是一次性开销，decode loop 是主要优化目标。

### 5.2 Backend 对比

| 指标 | `"dsl"` (fused) | `"3kernel"` |
|------|-----------------|-------------|
| 单次 kernel 调用延迟 | 快 **~1.46-1.49x** | 基准 |
| 端到端提升 | 仅 **~3%** | 基准 |
| 变长 history 支持 | 不支持 | 支持 (`seqused_k`) |
| SM100 (B200) | 不可用 | 唯一选项 |

Fused 单次调用快但端到端提升有限，因为 kernel 调用只是 decode loop 的一部分（还有 embedding lookup、FFN 等其他层操作）。

### 5.3 Dense vs Jagged

- 短序列: dense 路径比 jagged-native **快 ~5%**（jagged 的 offset 计算有额外开销）
- hist_len ≥ ~1000: jagged 开始胜出（dense 的 padding 浪费超过 offset 开销）

### 5.4 精度验证

#### 测试覆盖

| 测试 | Cases | 覆盖维度 |
|------|-------|----------|
| E2E (test_fwd) | 672 | dim × head_kv × beam_width × decode_nums × seqlen_context |
| K1 (test_context) | 144 | dim × head_kv × beam_width × seqlen_context |
| K2 (test_beam) | 384 | dim × head_kv × beam_width × decode_nums |

#### 容差标准

- **Output**: `kernel_diff <= 2 × pt_diff + fwd_atol` (FlashAttention-style, bf16 baseline)
- **LSE**: `abs_diff <= 1e-3`
- **E2E 回归**: top-1 exact match, log-prob delta < 0.15, top-K overlap ≥ 70%

---

## 6. 架构支持总览

| GPU | SM | K1 实现 | Fused Beam | Split-KV | 默认路径 |
|-----|-----|---------|-----------|----------|----------|
| A100 / L40 / L20 | SM8x | cp.async + mma.sync | 是 (256 threads) | 是 | fused + split-KV |
| H100 / H800 / H20 | SM90 | TMA + HGMMA | 是 (4-thread QK) | 是 | fused + split-KV |
| B200 / B300 | SM100 | tcgen05 + TMEM + CLC | **否** | 是 | 3-kernel + split-KV |
| RTX PRO6000 | SM120 | 继承 SM80 (cp.async) | 是 | 是 | fused + split-KV |

---

## 7. 已知问题与限制

1. **Split-KV + `seqused_k` 在 SM90 上 hang**: 已知问题，workaround 是自动 force `num_splits=1`。影响：长 context 下 K1 无法利用 split-KV 并行。
2. **Split-KV + `cu_seqlens_k` 不兼容**: Jagged 模式下 split-KV 的 partial-output geometry 假设 uniform Sk，强制 `ns=1`。
3. **非均匀 beam_width 不支持**: Kernel 断言 `k_beam.shape[1] == decode_nums × beam_width`，所有 hierarchy step 必须使用相同 beam_width。
4. **`beam_width` 上限**: 必须 `beam_width <= min(codebook_sizes)`，否则 propagate 无法正确选择 topK。
5. **Forward-only**: 不支持 backward（beam search 是推理阶段，不需要梯度）。
6. **`"dsl"` backend 忽略 `seqused_k`**: 如果误传 `seqused_k` 给 `"dsl"` 路径，会静默产生错误结果。`interface.py` 中有 assert 防护。

---

## 8. 来源与维护

- **上游仓库**: `ssh://git@gitlab-master.nvidia.com:12051/cjerry/gr-decode_atten.git`
- **锁定 commit**: `1c540f6` (2026-05-13 导入)
- **同步策略**: 按需拉取（on-demand），不定期同步
- **修改规则**: 不在本仓库直接修改 kernel 代码，所有改动必须先 upstream
- **引入 commit**: `23a03d4 [FEA] Beam search (#379)` (2026-05-19)
