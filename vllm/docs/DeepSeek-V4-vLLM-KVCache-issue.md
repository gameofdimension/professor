# [Bug] vLLM vastly over-estimates DeepSeek-V4 compressor-state KV cache (~33× real over-allocation)

**Model:** `deepseek-ai/DeepSeek-V4-Flash`  
**vLLM:** `0.22.1rc1.dev134+g6ece3bb8bfdb`  
**Serving args:** `tensor-parallel-size 8`, `--kv-cache-dtype fp8`, `--max-model-len 256000`, `--block-size 256`, `--enable-expert-parallel`  
**HW:** 8 × NVIDIA RTX PRO 5000 72GB (Blackwell)

---

## TL;DR

vLLM over-estimates DeepSeek-V4's **compressor state (fp32 transient accumulation scratch)** in the KV-cache budget by roughly **33×** (actually allocated ~14–15 GiB, true need only ~0.45 GiB). That estimate is **load-bearing** — it drives the startup feasibility check, `max_model_len` estimation, and max-concurrency estimation — and it is **actually allocated** (real over-allocation, not just a misleading log number), which caps token capacity.

Measured on the real model (TP=8): the KV pool gives compressor-fp32 **38%** of memory, while the MLA/indexer KV that actually stores tokens gets only **27%**.

---

## 1. Symptom & impact

Startup log (`bash v4flash.sh`):

```
Available KV cache memory: 37.93 GiB
GPU KV cache size: 2,497,693 tokens
Maximum concurrency for 256,000 tokens per request: 9.76x
```

That is **~16 KiB per token** (`37.93 GiB / 2,497,693`), much higher than expected. Investigation shows two problems:

1. **Conceptual mismatch:** sequence-length-independent transient scratch (compressor state, SWA) is amortized into a "per-token" figure, while the true per-token (scales with `seq_len`) is only ~3.4 KiB.
2. **Numerical inflation + real waste:** the compressor state is sized at ~2.47 GiB/request and is actually allocated ~14–15 GiB in HBM, while its real need is ~0.45 GiB.

### Consequences (all confirmed by code/measurement)

The inflated value is the output of `max_memory_usage_bytes` and is the **direct input** to these runtime decisions:

| Decision | Code location | Consequence |
|---|---|---|
| Startup feasibility check | `kv_cache_utils.py:691` `_check_enough_kv_cache_memory` | Under memory pressure, **falsely concludes "cannot fit" → refuses to start** (raises `ValueError`) |
| Max model-len estimation | `kv_cache_utils.py:740` `estimate_max_model_len` | **Forces a lower `max_model_len`** |
| Max concurrency / capacity | `kv_cache_utils.py:877` `get_max_concurrency_for_kv_cache_config` | **Under-estimates concurrency and token capacity** |
| Actual GPU allocation | `kv_cache_utils.py:935` `get_num_blocks` | **Actually reserves ~14.5 GiB for idle compressor state**, squeezing token capacity |

---

## 2. Root cause

Two independent bugs compound: `0.45 GiB × 5.5 × ~concurrency ≈ 24 GiB` (~54× reported attribution; ~33× after 2D packing).

### Factor 1 (single-point code bug; ~`compress_ratio`× per-request inflation)

The compressor-state spec does not carry `compress_ratio`, so the admission formula counts tokens instead of compression groups.

**① Spec omits `compress_ratio`** — `vllm/models/deepseek_v4/compressor.py:157`
```python
157  def get_kv_cache_spec(self, vllm_config):
158      return SlidingWindowMLASpec(            # <-- no compress_ratio=...
159          block_size=self.block_size, num_kv_heads=1, head_size=self.state_dim,
160          dtype=self.dtype, sliding_window=self.sliding_window, alignment=576)
```

**② Admission does not divide by `compress_ratio`** — `vllm/v1/kv_cache_interface.py:410`
```python
410  def max_admission_blocks_per_request(self, max_num_batched_tokens, max_model_len):
420      num_tokens = min(self.attention_chunk_size + max_num_batched_tokens, max_model_len)
423      return cdiv(num_tokens, self.block_size)   # token-count / groups-per-block -> off by compress_ratio
425  def max_memory_usage_bytes(self, vllm_config):
431      return max_blocks * self.page_size_bytes
```
A compressor block holds 4 compression groups (each group = `compress_ratio` tokens, i.e. 16 tokens/block for C4). The formula computes `8192 tokens / 4` instead of `8192 tokens / 16`, missing a `/compress_ratio` → block count inflated **4×** for C4 (and **~130×** for C128 where `compress_ratio=128`; aggregate **5.5×**).

### Factor 2 (structural sizing-model issue; ~concurrency× pool-level inflation)

`get_max_concurrency_for_kv_cache_config` (`kv_cache_utils.py:877`) and `get_num_blocks` (`:935`) size the pool as `concurrency × per_request`. This is correct for O(seq) attention KV, but wrong for a **batch-MNT-bounded** resource like the compressor: the whole batch shares `max_num_batched_tokens=8192`, so the total in-flight compression groups = `MNT / compress_ratio` (once), **not** `concurrency × (per-request MNT groups)`. The pool is therefore inflated by roughly the concurrency factor.

### Call stack (engine startup → inflated value → three consequences)

```
EngineCore.__init__
└─ engine/core.py:264   get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory)
   └─ kv_cache_utils.py:1945  get_kv_cache_configs
      ├─ :2038  _check_enough_kv_cache_memory(:691)            [refuse startup]
      │           └─ :817  max_memory_usage_bytes(:731)         <- inflated value summed here
      │           └─ :709  if needed > available: raise ValueError
      ├─ :2053  get_kv_cache_config_from_groups(:1236)
      │           └─ get_num_blocks(:935)                       [actual allocation]
      └─ :2074  _report_kv_cache_config(:1709)
                  └─ get_max_concurrency_for_kv_cache_config(:877)  [concurrency/token report]
                       └─ :886  num_layers × max_memory_usage_bytes(...)

Aggregation point: kv_cache_utils.py:731  max_memory_usage_bytes = Σ spec.max_memory_usage_bytes
        └─ compressor -> SlidingWindowSpec.max_memory_usage_bytes (kv_cache_interface.py:425)
              └─ max_admission_blocks_per_request(:410) -> cdiv(num_tokens, block_size)   <- BUG
Specs originate at: gpu_model_runner.py:6237 self.get_kv_cache_spec() -> each layer's compressor.get_kv_cache_spec (compressor.py:157)
```

---

## 3. Minimal reproduction

### 3a. CPU-only repro (no GPU, no weights) — uses vLLM's real spec classes & real `max_memory_usage_bytes()`

<details><summary><code>v4_kv_repro.py</code> (click to expand)</summary>

```python
#!/usr/bin/env python3
"""
Minimal repro: vLLM over-estimates DeepSeek-V4 compressor-state KV cache in
`max_memory_usage_bytes`, which is load-bearing for startup feasibility
(`check_enough_kv_cache_memory`), `estimate_max_model_len`, and
`get_max_concurrency`.

No GPU / no model weights needed — uses vLLM's real KVCacheSpec classes and the
real `max_memory_usage_bytes()` method, with a minimal mock VllmConfig.

Run:  python3 v4_kv_repro.py
"""
import types
import torch
from vllm.v1.kv_cache_interface import (MLAAttentionSpec, SlidingWindowMLASpec)

MML = 256_000          # max_model_len
MNT = 8_192            # max_num_batched_tokens (default)
BLOCK = 256            # cache block_size
HEAD = 512             # attention head_dim
IHEAD = 128            # indexer head_dim
WIN = 128              # sliding_window (SWA)
ALIGN = 576

# minimal VllmConfig stand-in (only the attrs max_memory_usage_bytes touches)
vllm_config = types.SimpleNamespace(
    model_config=types.SimpleNamespace(max_model_len=MML),
    parallel_config=types.SimpleNamespace(
        decode_context_parallel_size=1, prefill_context_parallel_size=1),
    scheduler_config=types.SimpleNamespace(max_num_batched_tokens=MNT),
)

def cdiv(a, b): return (a + b - 1) // b

# ---------- build DeepSeek-V4 cache specs (real classes, real params) ----------
# layer class counts: full=2, C4=21, C128=20  (43 decoder layers)
FULL, C4, C128 = 2, 21, 20
NL = FULL + C4 + C128

# SWA: every layer, uint8, block 64, 584 B/token (deepseek_v4 layout)
swa = [SlidingWindowMLASpec(block_size=64, num_kv_heads=1, head_size=HEAD,
        dtype=torch.uint8, sliding_window=WIN, model_version="deepseek_v4",
        alignment=ALIGN) for _ in range(NL)]

# compressed-attn main KV (fp8/uint8): C4 + C128 only
mla_c4  = [MLAAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=HEAD,
            dtype=torch.uint8, compress_ratio=4, model_version="deepseek_v4",
            cache_dtype_str="fp8", alignment=ALIGN) for _ in range(C4)]
mla_c128= [MLAAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=HEAD,
            dtype=torch.uint8, compress_ratio=128, model_version="deepseek_v4",
            cache_dtype_str="fp8", alignment=ALIGN) for _ in range(C128)]

# indexer KV (fp8/uint8): C4 only, head_dim 132
idx_c4 = [MLAAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=132,
            dtype=torch.uint8, compress_ratio=4, alignment=ALIGN) for _ in range(C4)]

# compressor state (fp32, SlidingWindow, NO compress_ratio in spec => default 1)
def comp_spec(cr):
    coff = 1 + (cr == 4)          # 2 for C4, 1 for C128
    sdim = 2 * coff * HEAD        # mla-compressor uses head_dim 512
    return SlidingWindowMLASpec(block_size=4 if cr == 4 else 8, num_kv_heads=1,
               head_size=sdim, dtype=torch.float32,
               sliding_window=coff * cr, alignment=ALIGN)
mla_cmp_c4   = [comp_spec(4)   for _ in range(C4)]
mla_cmp_c128 = [comp_spec(128) for _ in range(C128)]

def comp_idx_spec():
    sdim = 2 * 2 * IHEAD          # indexer-compressor uses head_dim 128
    return SlidingWindowMLASpec(block_size=4, num_kv_heads=1, head_size=sdim,
               dtype=torch.float32, sliding_window=8, alignment=ALIGN)
idx_cmp_c4 = [comp_idx_spec() for _ in range(C4)]

ALL = [("SWA", swa), ("MLA-main C4", mla_c4), ("MLA-main C128", mla_c128),
       ("indexer C4", idx_c4), ("mla-compress C4", mla_cmp_c4),
       ("mla-compress C128", mla_cmp_c128), ("idx-compress C4", idx_cmp_c4)]

# ---------- per-request memory via vLLM's real max_memory_usage_bytes ----------
print("="*74)
print("Per-request KV memory via vLLM real max_memory_usage_bytes(vllm_config)")
print("="*74)
print(f"{'cache':<20}{'#lay':>5}{'max_mem/req (B)':>18}{'per token':>12}")
print("-"*74)
total_per_req = 0
rows = {}
for name, specs in ALL:
    s = sum(x.max_memory_usage_bytes(vllm_config) for x in specs)
    total_per_req += s
    rows[name] = (len(specs), s)
    print(f"{name:<20}{len(specs):>5}{s:>18,}{s/MML:>12.1f}")
print("-"*74)
print(f"{'TOTAL per request':<20}{'':>5}{total_per_req:>18,}{total_per_req/MML:>12.1f}")
print(f"  => reported per-token = {total_per_req/MML/1024:.2f} KiB  (matches log's ~16 KiB)")

# ---------- derived concurrency / num_tokens ----------
BUDGET = int(37.93 * 1024**3)
conc = BUDGET / total_per_req
print(f"\nBudget={BUDGET:,} B  ->  derived max_concurrency = budget/per_req = {conc:.2f}x")
print(f"  => reported num_tokens = concurrency * max_model_len = {conc*MML:,.0f}"
      f"  (matches log's 2,497,693)")

# ---------- TRUE in-flight compressor need (what it SHOULD cost) ----------
print("\n" + "="*74)
print("TRUE concurrent compressor-state need (hand-computed physics)")
print("="*74)
def true_comp(cr, sdim, nlayers):
    groups = MNT // cr
    return groups * sdim * 4 * nlayers       # fp32
t_c4   = true_comp(4,   2*2*HEAD, C4)
t_c128 = true_comp(128, 2*1*HEAD, C128)
t_idx  = true_comp(4,   2*2*IHEAD, C4)
true_cmp = t_c4 + t_c128 + t_idx
print(f"  in-flight C4 groups  = MNT/4   = {MNT//4}")
print(f"  in-flight C128 groups= MNT/128 = {MNT//128}")
print(f"  mla-compress C4   : {t_c4:>14,} B")
print(f"  mla-compress C128 : {t_c128:>14,} B")
print(f"  idx-compress C4   : {t_idx:>14,} B")
print(f"  TRUE compressor total = {true_cmp:>14,} B ({true_cmp/1e9:.2f} GB)")

reported_cmp = (rows["mla-compress C4"][1] + rows["mla-compress C128"][1]
                + rows["idx-compress C4"][1])
print(f"\n  vLLM-reported compressor per-request = {reported_cmp:,} B ({reported_cmp/1e9:.2f} GB)")
print(f"  OVER-COUNT factor (per request)       = {reported_cmp/true_cmp:.1f}x")
print(f"  (then x{conc:.1f} concurrency in the pool view => "
      f"{reported_cmp*conc/1e9:.0f} GB attributed vs {true_cmp/1e9:.2f} GB real)")

# ---------- what the per-token SHOULD be ----------
linear = rows["MLA-main C4"][1] + rows["MLA-main C128"][1] + rows["indexer C4"][1]
print("\n" + "="*74)
print("Bottom line")
print("="*74)
print(f"  Truly seq-proportional per-token (MLA-main + indexer) = {linear/MML/1024:.2f} KiB")
print(f"  + SWA (window, per-req constant) amortized            = {rows['SWA'][1]/MML/1024:.2f} KiB")
print(f"  + compressor TRUE amortized                           = {true_cmp/MML/1024:.2f} KiB")
print(f"  = TRUE per-token                                      ≈ {(linear+rows['SWA'][1]+true_cmp)/MML/1024:.2f} KiB")
print(f"  vLLM REPORTS per-token                                ≈ {total_per_req/MML/1024:.2f} KiB  (inflated)")
```

</details>

Output (matches the log):
```
cache                #lay   max_mem/req (B)   per token
mla-compress C4        21     1,414,107,072      5523.9   <- fp32 scratch, inflated
mla-compress C128      20       683,562,240      2670.2
MLA-main C4            21       689,472,000      2693.2
idx-compress C4        21       372,133,440      1453.6
SWA                    43       210,899,520       823.8
indexer C4             21       181,440,000       708.8
MLA-main C128          20        23,040,000        90.0
TOTAL per request         3,574,654,272     13963.5   -> reported ~16 KiB

vLLM-reported compressor/req = 2.47 GB
TRUE concurrent compressor  = 0.45 GB
OVER-COUNT (per request)    = 5.5x   (C4 layer 4x, C128 layer ~130x)
TRUE per-token ≈ 5.91 KiB   vs   vLLM REPORTS 13.64 KiB
```

### 3b. Real-model verification (wraps `get_kv_cache_configs` during a real TP=8 init)

<details><summary><code>v4_verify/sitecustomize.py</code> (click to expand)</summary>

```python
"""Verification hooks (auto-imported via PYTHONPATH). Wraps vLLM internals during
a REAL model init to dump: (1) real per-spec page sizes, (2) real num_blocks,
(3) ACTUAL per-tensor KV allocation summed by cache type."""
import os
from collections import defaultdict

def _install():
    try:
        import torch
        from vllm.v1.core import kv_cache_utils as m
        from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec
    except Exception:
        return False

    def kind(spec):
        if isinstance(spec, MLAAttentionSpec):
            return f"MLAAttentionSpec(fp8?) dtype={spec.dtype} hs={spec.head_size} cr={getattr(spec,'compress_ratio',1)}"
        if isinstance(spec, SlidingWindowMLASpec):
            t = "compressor-fp32" if spec.dtype == torch.float32 else "SWA-fp8"
            return f"SlidingWindowMLASpec[{t}] dtype={spec.dtype} hs={spec.head_size}"
        return type(spec).__name__

    if not getattr(m.get_kv_cache_configs, "_v4verify", False):
        _real_gkcc = m.get_kv_cache_configs
        def wrapped_gkcc(vllm_config, kv_cache_specs, available_memory):
            configs = _real_gkcc(vllm_config, kv_cache_specs, available_memory)
            try:
                specs = kv_cache_specs[0]
                ps = defaultdict(set)
                for n, s in specs.items():
                    ps[kind(s)].add(s.page_size_bytes)
                m.logger.warning("VERIFY-A real page_size_bytes by spec kind:")
                for k, v in ps.items():
                    m.logger.warning("VERIFY-A   %-55s page=%s", k, sorted(v))
                kc = configs[0]
                m.logger.warning("VERIFY-B num_blocks=%s  available=%s B", kc.num_blocks,
                                 available_memory[0])
                agg = defaultdict(int); total = 0
                for t in kc.kv_cache_tensors:
                    total += t.size
                    for n in t.shared_by:
                        s = specs.get(n)
                        kk = "compressor-fp32" if (isinstance(s, SlidingWindowMLASpec)
                                and s.dtype == torch.float32) else \
                             "SWA-fp8" if isinstance(s, SlidingWindowMLASpec) else \
                             "MLA/indexer-fp8" if isinstance(s, MLAAttentionSpec) else "?"
                        agg[kk] += t.size // max(1, len(t.shared_by))
                m.logger.warning("VERIFY-C ACTUAL allocation by cache type (total=%.2f GiB):", total/2**30)
                for k, v in sorted(agg.items(), key=lambda x: -x[1]):
                    m.logger.warning("VERIFY-C   %-20s %8.2f GiB  %5.1f%%", k, v/2**30, 100*v/total)
            except Exception as e:
                m.logger.warning("VERIFY dump failed: %r", e)
            return configs
        wrapped_gkcc._v4verify = True
        m.get_kv_cache_configs = wrapped_gkcc
        m.logger.warning("VERIFY: wrapped get_kv_cache_configs")
    return True

if not _install():
    import builtins as _b
    _ri = _b.__import__
    def _w(name, *a, **k):
        m = _ri(name, *a, **k)
        if name.startswith("vllm"):
            _install()
        return m
    _b.__import__ = _w
```

</details>

Run:
```bash
PYTHONPATH=/tmp/v4_verify python3 -c "from vllm import LLM; LLM(model='/path/DeepSeek-V4-Flash', kv_cache_dtype='fp8', tensor_parallel_size=8, max_model_len=256000, block_size=256, trust_remote_code=True, enable_expert_parallel=True, enforce_eager=True, tokenizer_mode='deepseek_v4')"
```

Measured output:
```
VERIFY-A real page_size_bytes by spec kind:
  SlidingWindowMLASpec[SWA-fp8]        dtype=uint8   hs=512  page=[37440]
  MLAAttentionSpec dtype=uint8 hs=132 cr=4                    page=[8640]     (indexer)
  SlidingWindowMLASpec[compressor-fp32] dtype=float32 hs=512  page=[8640]     (idx-compress)
  MLAAttentionSpec dtype=uint8 hs=512 cr=4                    page=[37440]    (MLA-main C4, padded to 37440)
  SlidingWindowMLASpec[compressor-fp32] dtype=float32 hs=2048 page=[37440]    (mla-compress C4)
  MLAAttentionSpec dtype=uint8 hs=512 cr=128                  page=[1728]     (MLA-main C128)
  SlidingWindowMLASpec[compressor-fp32] dtype=float32 hs=1024 page=[37440]    (mla-compress C128)
VERIFY-B num_blocks=40471  available=42566892135 B
VERIFY-C ACTUAL allocation by cache type (total=39.64 GiB):
  compressor-fp32         15.06 GiB   38.0%   <- actually allocated
  SWA-fp8                 13.41 GiB   33.8%
  MLA/indexer-fp8         10.72 GiB   27.0%   <- the only part that actually stores tokens
```
→ compressor fp32 is actually allocated **~33× more** than its 0.45 GiB need — **confirmed real over-allocation**.

---

## 4. Proposed fixes

### Fix A (Factor 1) — preferred; single-point, low-risk, can be its own PR

Make the compressor admission convert tokens↔compression-groups correctly. Two equivalent options:

- **A1:** in `compressor.py:158` pass `compress_ratio=self.compress_ratio` to the `SlidingWindowMLASpec`, and in `SlidingWindowMLASpec.max_admission_blocks_per_request` divide `num_tokens` by `compress_ratio` first (count by group): `cdiv(num_tokens // compress_ratio, block_size)`.
- **A2:** express the compressor spec's `block_size` in token units (`block_size × compress_ratio`) so the existing `cdiv(num_tokens, block_size)` is already correct (requires adjusting the tensor layout — larger change).

Expected effect: per-request compressor drops 2.47 GiB → ~0.45 GiB (5.5×); reported per-token ~16 KiB → ~6 KiB; `num_tokens` 2.49M → ~6M+. No impact on attention KV / SWA.

### Fix B (Factor 2) — structural; suggested follow-up

Size batch-MNT-bounded transient caches (compressor, SWA) at the **pool/batch level** rather than `per-request × concurrency`. MNT is a batch-shared budget; total in-flight groups ≤ `MNT/compress_ratio`, independent of concurrency. Requires reworking how `get_max_concurrency` / `get_num_blocks` size this class of cache. Expected to push capacity toward the physical limit (~8M tokens).

### Fix C (precision) — orthogonal, separate discussion

The compressor state is hard-coded `dtype=torch.float32` (`compressor.py:215`). It only accumulates `compress_ratio` (4/128) tokens before flushing, so its long-range precision pressure is far smaller than for Mamba SSM state — **bf16 is very likely safe** (cf. vLLM's existing `mamba_ssm_cache_dtype` fp16/bf16 precedent). Halves both true need and allocation, but needs validation of the `kv_state/score_state` normalization under long contexts.

> Priority: **A > B > C**. A is a clean bugfix; B is a design improvement; C is an independent optimization.

---

## 5. Theoretical memory derivation

### 5.1 Model structure (hybrid compressed attention, not MLA)

`config.json`: `num_hidden_layers=43`, `head_dim=512`, `num_key_value_heads=1`, `q_lora_rank=1024`, `qk_rope_head_dim=64`, `sliding_window=128`, `compress_ratios=[0,0,(4,128)×20,4,0]`.

Layers classified by `compress_ratio = max(1, compress_ratios[i])`: **full×2 / C4A×21 / C128A×20**. Each decoder layer carries several independent caches:

| cache | dtype | nature | scales with seq_len? |
|---|---|---|---|
| SWA sliding-window KV | uint8 | window-bounded (window=128) | No (per-request constant) |
| compressed-attn main KV (C4A/C128A) | uint8 | stores compressed KV | **Yes** (÷ compress_ratio) |
| indexer sparse-index KV (C4A only) | uint8 | stores compressed keys | **Yes** (÷ 4) |
| compressor state ×2/1 | **float32** | in-flight compression-group accumulator scratch | No (batch-MNT-bounded) |

> `num_key_value_heads=1` ⇒ the KV latent is **not** TP-sharded; all 8 ranks store a full copy (TP does not reduce KV cache).

### 5.2 Truly seq-proportional per-token (≈ 3.4 KiB)

This is the only part that should be measured "per token". `per token = head_dim(512 B) / compress_ratio × #layers`:

| cache | per layer B/token | layers | total B/token |
|---|---|---|---|
| compressed main KV C4A | 512/4 = 128 | 21 | 2688 |
| indexer C4A | 132/4 = 33 | 21 | 709 |
| compressed main KV C128A | 512/128 = 4 | 20 | 80 |
| **true per-token subtotal** | | | **≈ 3.4 KiB** |

### 5.3 True concurrent compressor-state footprint (≈ 0.45 GiB)

`footprint = in_flight_groups × state_dim × fp32 × #layers`. In-flight groups are bounded by the batch `MNT=8192`:

```
in-flight C4 groups   = 8192/4   = 2048
in-flight C128 groups = 8192/128 = 64
state_dim = 2 × coff × head_dim   (2 = kv_state+score_state; coff = 1+(cr==4))
```

| compressor | in-flight groups | state_dim | fp32 | layers | footprint |
|---|---|---|---|---|---|
| mla-compress C4 | 2048 | 2048 | 4 | 21 | 2048×2048×4×21 = **352 MB** |
| idx-compress C4 | 2048 | 512 | 4 | 21 | 2048×512×4×21 = **88 MB** |
| mla-compress C128 | 64 | 1024 | 4 | 20 | 64×1024×4×20 = **5 MB** |
| **true total** | | | | | **≈ 445 MB ≈ 0.45 GiB** |

### 5.4 vLLM's inflation chain

```
true 0.45 GiB
   │  × Factor 1 (compress_ratio not divided, aggregate 5.5×)
   ▼
per-request 2.47 GiB        <- what max_memory_usage_bytes reports
   │  × Factor 2 (per-request × concurrency, ~concurrency×)
   ▼
pool attribution ~24 GiB (54×)   <- reported view
   │  (2D layer-tuple packing is more efficient)
   ▼
actual allocation ~15 GiB (33×)  <- measured VERIFY-C
```

| quantity | value | source |
|---|---|---|
| True per-token (linear) | ~3.4 KiB | §5.2 |
| True concurrent compressor footprint | 0.45 GiB | §5.3 |
| vLLM reported per-token | ~16 KiB | startup log |
| vLLM reported compressor/request | 2.47 GiB | `v4_kv_repro.py` |
| **compressor actual allocation** | **~15 GiB (38%)** | measured `VERIFY-C` |
| Actual over-allocation factor | **~33×** | 15 GiB / 0.45 GiB |
| Share going to token-storing KV | 27% (~10.7 GiB) | measured `VERIFY-C` |

---

## Appendix — environment & code index

- **Environment:** vLLM `0.22.1rc1.dev134+g6ece3bb8bfdb`, DeepSeek-V4-Flash, 8 × RTX PRO 5000 72GB, TP=8.
- Note: the real-model verification used `enforce_eager` and did not pin `max_num_batched_tokens`, so the concurrency figure does not match the production log (9.76×) exactly — but the **allocation share (38%) is determined by page-size × grouping and is independent of MNT/cudagraph**, so the conclusion is unaffected.
- Measured real page sizes: MLA-main C4 = 37440, MLA-main C128 = 1728, indexer = 8640, SWA = 37440, all three compressor variants = 37440 (all padded to the 37440 bucket).

**Code index**
- Root cause ①: `vllm/models/deepseek_v4/compressor.py:157-166`
- Root cause ②: `vllm/v1/kv_cache_interface.py:410-431`
- Aggregation point: `vllm/v1/core/kv_cache_utils.py:731-737`
- Consequence exits: `vllm/v1/core/kv_cache_utils.py:691` (refuse startup) / `:740` (max_model_len) / `:877-894` (concurrency) / `:935` (actual allocation)
- Engine entry: `vllm/engine/core.py:264` → `vllm/v1/core/kv_cache_utils.py:1945` (`get_kv_cache_configs`)
