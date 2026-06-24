# DeepSeek-V4 在 vLLM 中 KV cache 显存被严重高估（compressor state over-count）

> 模型：`deepseek-ai/DeepSeek-V4-Flash`　vLLM：`0.22.1rc1.dev134+g6ece3bb8bfdb`　TP=8　`--kv-cache-dtype fp8`　`max_model_len=256000`　`block_size=256`
> 8× NVIDIA RTX PRO 5000 72GB (Blackwell)

---

## TL;DR

vLLM 把 DeepSeek-V4 的 **compressor state（fp32 瞬态累加 scratch）** 在 KV cache 预算里高估了约 **33 倍**（实际占用 ~14–15 GiB，真实需求仅 ~0.45 GiB）。这个高估值是 **load-bearing** 的——它直接决定启动可行性校验、`max_model_len` 估计、最大并发估计，**并且实际把 GPU 显存真分配出去了**（over-allocate，不是只上报虚高），导致 token 容量被压低。

实测（真实模型 TP=8 实跑）：KV pool 里 compressor-fp32 实际占 **38%**，而真正存 token 的 MLA/indexer KV 只拿到 **27%**。

---

## 1. 现象与问题

启动日志（`bash v4flash.sh`）：

```
Available KV cache memory: 37.93 GiB
GPU KV cache size: 2,497,693 tokens
Maximum concurrency for 256,000 tokens per request: 9.76x
```

折算 **每 token KV cache ≈ 16 KiB**（`37.93 GiB / 2,497,693`），显著高于预期。

排查发现这个数字有两类问题：

1. **概念错配**：把不随序列增长的瞬态 scratch（compressor state、SWA）摊进了「每 token」，而真正的 per-token（随 `seq_len` 线性增长）只有 ~3.4 KiB。
2. **数值高估 + 实际浪费**：compressor state 被算成每请求 ~2.47 GiB，实际在 HBM 里真分到了 ~14–15 GiB，而它真实只需 ~0.45 GiB。

### 后果（均为实测/代码确认）

这个高估值是 `max_memory_usage_bytes` 的输出，是下列运行期决策的**直接输入**：

| 决策 | 代码位置 | 后果 |
|---|---|---|
| 启动可行性校验 | `kv_cache_utils.py:691` `_check_enough_kv_cache_memory` | 显存紧张时**误判放不下 → 拒启动**（raise ValueError） |
| 最大序列长度估计 | `kv_cache_utils.py:740` `estimate_max_model_len` | **被迫压低 max_model_len** |
| 最大并发/容量 | `kv_cache_utils.py:877` `get_max_concurrency_for_kv_cache_config` | **并发与 token 容量估低** |
| 实际显存分配 | `kv_cache_utils.py:935` `get_num_blocks` | **真的把 ~14.5 GiB 分给空转的 compressor**，挤占 token 容量 |

---

## 2. 原因

两层独立 bug 叠加：`0.45 GiB × 5.5 × ~并发 ≈ 24 GiB`（上报归因 ~54×；2D 打包后实际 ~33×）。

### Factor 1（单点代码 bug，per-request 高估 ~compress_ratio 倍）

compressor state 的 spec 没有携带 `compress_ratio`，导致 admission 公式按 token 计数而非压缩组计数。

**① spec 漏传 `compress_ratio`** —— `vllm/models/deepseek_v4/compressor.py:157`
```python
157  def get_kv_cache_spec(self, vllm_config):
158      return SlidingWindowMLASpec(            # ← 没有 compress_ratio=...
159          block_size=self.block_size, num_kv_heads=1, head_size=self.state_dim,
160          dtype=self.dtype, sliding_window=self.sliding_window, alignment=576)
```

**② admission 漏除 compress_ratio** —— `vllm/v1/kv_cache_interface.py:410`
```python
410  def max_admission_blocks_per_request(self, max_num_batched_tokens, max_model_len):
420      num_tokens = min(self.attention_chunk_size + max_num_batched_tokens, max_model_len)
423      return cdiv(num_tokens, self.block_size)   # token数 ÷ 每块组数 → 多算 compress_ratio 倍
425  def max_memory_usage_bytes(self, vllm_config):
431      return max_blocks * self.page_size_bytes
```
compressor 一块装 4 个压缩组（每组 = `compress_ratio` 个 token，C4 即 16 token/块）；公式却按 `8192 token / 4` 算 block，少除了 compress_ratio=4 → block 数高估 **4×**（C128 层因 compress_ratio=128 高达 **~130×**，聚合 **5.5×**）。

### Factor 2（结构性 sizing 模型问题，池级再高估 ~并发倍）

`get_max_concurrency_for_kv_cache_config`(`kv_cache_utils.py:877`) 与 `get_num_blocks`(`:935`) 的整套 sizing 假设 **池 = 并发 × per-request**。这对 O(seq) 的 attention KV 正确，但对 compressor 这种 **batch-MNT 限长**的资源是错的：整个 batch 一步共享 `max_num_batched_tokens=8192`，在飞压缩组总数 = `MNT/compress_ratio`（一次），并非「并发 × 每请求 MNT 组」。于是池级再高估约并发倍。

### 调用栈（engine 启动 → 高估值 → 三个后果）

```
EngineCore.__init__
└─ engine/core.py:264   get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory)
   └─ kv_cache_utils.py:1945  get_kv_cache_configs
      ├─ :2038  _check_enough_kv_cache_memory(:691)            【拒启动】
      │           └─ :817  max_memory_usage_bytes(:731)         ← 高估值在此累加
      │           └─ :709  if needed > available: raise ValueError
      ├─ :2053  get_kv_cache_config_from_groups(:1236)
      │           └─ get_num_blocks(:935)                       【实际分配】
      └─ :2074  _report_kv_cache_config(:1709)
                  └─ get_max_concurrency_for_kv_cache_config(:877)  【并发/token 上报】
                       └─ :886  num_layers × max_memory_usage_bytes(...)

累加点：kv_cache_utils.py:731  max_memory_usage_bytes = Σ spec.max_memory_usage_bytes
        └─ compressor 走 SlidingWindowSpec.max_memory_usage_bytes (kv_cache_interface.py:425)
              └─ max_admission_blocks_per_request(:410) → cdiv(num_tokens, block_size)  ← BUG
spec 来源：gpu_model_runner.py:6237 self.get_kv_cache_spec() → 各层 compressor.get_kv_cache_spec(compressor.py:157)
```

---

## 3. 最小复现

**`/tmp/v4_kv_repro.py`** —— 纯 CPU、无需权重，直接调用 vLLM 真实的 `MLAAttentionSpec` / `SlidingWindowMLASpec` 类与真实的 `max_memory_usage_bytes()` 方法，构造 V4 的 cache 结构。

```bash
python3 /tmp/v4_kv_repro.py
```

关键输出（与日志吻合）：
```
cache                #lay   max_mem/req (B)   per token
mla-compress C4        21     1,414,107,072      5523.9   ← fp32 scratch，被高估
mla-compress C128      20       683,562,240      2670.2
MLA-main C4            21       689,472,000      2693.2
idx-compress C4        21       372,133,440      1453.6
SWA                    43       210,899,520       823.8
indexer C4             21       181,440,000       708.8
MLA-main C128          20        23,040,000        90.0
TOTAL per request         3,574,654,272     13963.5   → 上报 ~16 KiB ✓

vLLM-reported compressor/req = 2.47 GB
TRUE concurrent compressor  = 0.45 GB
OVER-COUNT (per request)    = 5.5×   (C4 层 4×, C128 层 ~130×)
TRUE per-token ≈ 5.91 KiB   vs   vLLM REPORTS 13.64 KiB
```

**真实模型实证**（`/tmp/v4_verify/sitecustomize.py` 包装 `get_kv_cache_configs`，真实 TP=8 init）：
```
VERIFY-C ACTUAL allocation by cache type (total=39.64 GiB):
  compressor-fp32         15.06 GiB   38.0%   ← 真分配出去的
  SWA-fp8                 13.41 GiB   33.8%
  MLA/indexer-fp8         10.72 GiB   27.0%   ← 真正存 token 的
```
→ compressor 实际 over-allocate **~33×**（15 GiB vs 0.45 GiB），**确认是真浪费显存**。

---

## 4. 提议修复方案

### Fix A（Factor 1，首选，单点、低风险、可独立 PR）

让 compressor 的 admission 正确换算 token↔压缩组。两种等价实现：

- **A1**：`compressor.py:158` 给 `SlidingWindowMLASpec` 传入 `compress_ratio=self.compress_ratio`，并在 `SlidingWindowMLASpec.max_admission_blocks_per_request` 里先把 `num_tokens` 除以 `compress_ratio`（即按组计数）：`cdiv(num_tokens // compress_ratio, block_size)`。
- **A2**：让 compressor spec 的 `block_size` 以 token 为单位表达（`block_size × compress_ratio`），使现有 `cdiv(num_tokens, block_size)` 直接正确（需同步调整 tensor 布局，改动更大）。

预期效果：per-request compressor 从 2.47 GiB → ~0.45 GiB（降 5.5×）；上报 per-token 从 ~16 KiB → ~6 KiB；`num_tokens` 从 2.49M → ~6M+。对 attention KV / SWA 无影响。

### Fix B（Factor 2，结构性，建议 follow-up）

对 batch-MNT 限长的瞬态 cache（compressor、SWA）按**池/batch 级**而非 per-request × 并发计尺寸。MNT 是 batch 共享预算，在飞组总数 ≤ `MNT/compress_ratio`，与并发数无关。需重构 `get_max_concurrency` / `get_num_blocks` 对这类 cache 的尺寸模型。预期可把容量进一步推到接近物理上限（~8M tokens）。

### Fix C（精度，正交、另议）

compressor state 当前写死 `dtype=torch.float32`（`compressor.py:215`）。它只累加 `compress_ratio`（4/128）个 token 就 flush，长程精度压力远小于 Mamba SSM state，**大概率可降到 bf16**（参考 vLLM 已有 `mamba_ssm_cache_dtype` 可选 fp16/bf16 的先例）。降精度可将真实需求与分配再砍半，但需验证长序列下 `kv_state/score_state` 归一化的数值稳定性。

> 优先级：**A > B > C**。A 是干净的 bugfix，B 是设计改进，C 是独立优化。建议 issue 以 A 为主、B/C 作为延伸讨论。

---

## 5. 理论占用计算过程

### 5.1 模型结构（非 MLA，是混合压缩注意力）

`config.json`：`num_hidden_layers=43`，`head_dim=512`，`num_key_value_heads=1`，`q_lora_rank=1024`，`qk_rope_head_dim=64`，`sliding_window=128`，`compress_ratios=[0,0,(4,128)×20,4,0]`。

每层按 `compress_ratio = max(1, compress_ratios[i])` 分类：**full×2 / C4A×21 / C128A×20**。每个 decoder 层挂多份独立 cache：

| cache | dtype | 性质 | 是否随 seq_len 增长 |
|---|---|---|---|
| SWA 滑窗 KV | uint8 | 滑窗限长（window=128） | 否（每请求常数） |
| 压缩注意力主 KV（C4A/C128A） | uint8 | 存压缩后 KV | **是**（按 1/compress_ratio） |
| indexer 稀疏索引 KV（仅 C4A） | uint8 | 存压缩 key | **是**（按 1/4） |
| compressor state ×2/1 | **float32** | 在飞压缩组累加 scratch | 否（batch-MNT 限长） |

> `num_key_value_heads=1` ⇒ KV 隐变量不被 TP 切分，8 卡各存完整副本（TP 不省 KV）。

### 5.2 真正随序列线性的 per-token（≈3.4 KiB）

这部分才是「每 token KV cache」的本体。`per token = head_dim(512B) / compress_ratio × 层数`：

| cache | 单层 B/token | 层数 | 合计 B/token |
|---|---|---|---|
| 压缩主 KV C4A | 512/4 = 128 | 21 | 2688 |
| indexer C4A | 132/4 = 33 | 21 | 709 |
| 压缩主 KV C128A | 512/128 = 4 | 20 | 80 |
| **真·per-token 小计** | | | **≈ 3.4 KiB** |

### 5.3 compressor state 真实并发占用（≈0.45 GiB）

`占用 = 在飞组数 × state_dim × fp32 × 层数`。在飞组数被 batch `MNT=8192` 卡住：

```
在飞 C4 组   = 8192/4   = 2048
在飞 C128 组 = 8192/128 = 64
state_dim    = 2 × coff × head_dim   （2 = kv_state+score_state；coff = 1+(cr==4)）
```

| 压缩器 | 在飞组 | state_dim | fp32 | 层数 | 占用 |
|---|---|---|---|---|---|
| mla-compress C4 | 2048 | 2048 | 4 | 21 | 2048×2048×4×21 = **352 MB** |
| idx-compress C4 | 2048 | 512 | 4 | 21 | 2048×512×4×21 = **88 MB** |
| mla-compress C128 | 64 | 1024 | 4 | 20 | 64×1024×4×20 = **5 MB** |
| **真实合计** | | | | | **≈ 445 MB ≈ 0.45 GiB** |

### 5.4 vLLM 的膨胀链路

```
真实 0.45 GiB
   │  × Factor 1 (compress_ratio 漏除, 聚合 5.5×)
   ▼
per-request 2.47 GiB      ← max_memory_usage_bytes 报的
   │  × Factor 2 (per-request × 并发, ~并发倍)
   ▼
池级归因 ~24 GiB (54×)    ← 上报口径
   │  (2D 层元组打包实际更省)
   ▼
实际分配 ~15 GiB (33×)    ← 实测 VERIFY-C
```

| 量 | 值 | 来源 |
|---|---|---|
| 真·per-token（线性） | ~3.4 KiB | §5.2 |
| 真·compressor 并发占用 | 0.45 GiB | §5.3 |
| vLLM 上报 per-token | ~16 KiB | 日志 |
| vLLM 上报 compressor/请求 | 2.47 GiB | `/tmp/v4_kv_repro.py` |
| **compressor 实际分配** | **~15 GiB (38%)** | 实测 `VERIFY-C` |
| 实际 over-allocate 倍数 | **~33×** | 15 GiB / 0.45 GiB |
| 真正存 token 的 KV 占比 | 27%（~10.7 GiB） | 实测 `VERIFY-C` |

---

## 附：验证方法与环境

- **环境**：vLLM `0.22.1rc1.dev134+g6ece3bb8bfdb`，DeepSeek-V4-Flash（`/warehouse/DeepSeek-V4-Flash`），8× RTX PRO 5000 72GB，TP=8。
- **最小复现**：`/tmp/v4_kv_repro.py`（CPU，即开即跑）。
- **真实实证 patch**：`/tmp/v4_verify/sitecustomize.py`，用法：
  ```bash
  VLLM_KV_CACHE_DEBUG=1 PYTHONPATH=/tmp/v4_verify python3 -c "from vllm import LLM; LLM(model='...', kv_cache_dtype='fp8', tensor_parallel_size=8, max_model_len=256000, block_size=256, trust_remote_code=True, enable_expert_parallel=True, enforce_eager=True, tokenizer_mode='deepseek_v4')"
  ```
  （注：本验证为提速用 `enforce_eager` 且未设 `max_num_batched_tokens`，故并发数与日志（9.76x）不严格一致；但**实际分配占比（38%）由 page×分组决定，与 MNT/cudagraph 无关**，结论不受影响。）

### 相关代码位置索引
- 根因①：`vllm/models/deepseek_v4/compressor.py:157-166`
- 根因②：`vllm/v1/kv_cache_interface.py:410-431`
- 累加点：`vllm/v1/core/kv_cache_utils.py:731-737`
- 后果出口：`vllm/v1/core/kv_cache_utils.py:691`(拒启动) / `:740`(max_model_len) / `:877-894`(并发) / `:935`(实际分配)
- 真实 page_size（实测）：MLA-main C4=37440、MLA-main C128=1728、indexer=8640、SWA=37440、compressor 三档均=37440（均被 pad 到 37440 桶）
