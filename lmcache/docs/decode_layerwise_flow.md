# Decode 阶段 Layerwise 流程分析

> 本文档分析 `LMCACHE_USE_LAYERWISE=true` 时，Decode 阶段的执行流程，
> 重点对比与 Prefill 阶段的差异。关联文档：[prefill_layerwise_flow.md](prefill_layerwise_flow.md)。

---

## 核心结论：默认情况下 Decode 阶段 Layerwise 是空操作

**默认配置下（`save_decode_cache=False`），decode 阶段的 4 个 connector 方法全部退化为 no-op。**
LMCache 的 layerwise 流水线仅在 prefill 阶段活跃。

差异不在 worker 侧的 4 个方法内部，而在 **scheduler 侧的元数据构建**——
通过 `load_spec` / `save_spec` 的有无来控制。

---

## 差异对比：Prefill vs Decode

### 1. Scheduler 侧：`lookup()` 根本不被调用

**Prefill 路径（新请求）：**

```
vLLM scheduler 处理 WAITING 队列
  → if request.num_computed_tokens == 0:        # scheduler.py:614
      → connector.get_num_new_matched_tokens()   # 触发 lookup()
        → lookup_client.lookup(tokens, pin=True) # 查询缓存命中
        → 创建 LoadSpec(can_load=False)           # vllm_v1_adapter.py:1436
      → connector.update_state_after_alloc()      # vllm_v1_adapter.py:1452
        → load_specs[req_id].can_load = True      # line 1523
```

**Decode 路径（运行中请求）：**

```
vLLM scheduler 处理 RUNNING 队列
  → (不调用任何 connector 方法)
  → 直接 allocate_slots() 继续推理
```

Decode 请求的 `num_computed_tokens > 0`，所以 scheduler 的 WAITING 路径根本不触发。
在 `build_connector_meta` 处理 cached_reqs 时（`vllm_v1_adapter.py:1658-1659`）：

```python
load_spec = self.load_specs.pop(req_id, None)  # 返回 None — 从未被设置
```

### 2. `RequestTracker.is_decode_phase` 的判定

`vllm_v1_adapter.py:271-272`：

```python
if len(new_token_ids) == 1:
    self.is_decode_phase = True
```

Decode 每步只生成 1 个 token，所以 `is_decode_phase` 在首次 decode 步骤后变为 `True` 并永远保持。

### 3. `ReqMeta.from_request_tracker()` 的关键分流点

`vllm_v1_adapter.py:340-348`：

```python
skip_save = tracker.disagg_spec is None and (
    tracker.skip_save                                          # 用户显式跳过
    or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)  # 未达 chunk 边界
    or (tracker.is_decode_phase and not save_decode_cache)    # ← decode 阶段默认跳过
    or request_skip                                            # 请求级跳过
)

if skip_save and load_spec is None:   # ← 关键退出点
    return None                        # 不生成元数据，请求不出现在 connector metadata 中
```

对于 decode 请求：

- `is_decode_phase = True`（line 272）
- `save_decode_cache = False`（默认，`config.py:102-106`）
- `load_spec = None`（scheduler 从未设置）
- **结果**：`skip_save = True` + `load_spec is None` → **`return None`** → 请求被排除在 connector metadata 之外

### 4. Worker 侧 4 个方法在 Decode 下的行为

| 方法 | Prefill | Decode（默认） |
|---|---|---|
| `start_load_kv` | 遍历请求，`load_spec.can_load=True` 触发 `retrieve_layer()` generator | 所有请求 `load_spec is None`，循环被 `continue` 跳过 → `layerwise_retrievers` 为空列表 |
| `wait_for_layer_load` | 对每个 retriever 调用 `next()`，推进逐层加载 | `self.layerwise_retrievers` 为空 → 循环体不执行，仅 `current_layer += 1` |
| `save_kv_layer` | `use_layerwise=True` + `can_save=True` → 创建/推进 `store_layer()` generator | 请求不在 metadata 中（`from_request_tracker` 返回了 `None`）→ 循环无请求可处理 |
| `wait_for_save` | 推进 storer 最后一步 + `lookup_unpin()` | `layerwise_storer` pop 为 `None` → 仅执行 `lookup_unpin()` |

**简单来说：4 个方法全部走一遍，但内部没有有效请求需要处理，开销极小。**

---

## 完整时序对比图

```
                         Prefill 阶段                              Decode 阶段 (默认)
                         ───────────                              ──────────────────

Scheduler:
  队列                    WAITING 队列                             RUNNING 队列
  num_computed_tokens     0                                        > 0
  lookup()                ✅ 调用, 返回命中 token 数               ❌ 不调用
  get_num_new_matched     ✅ 触发                                  ❌ 不触发
  update_state_after_alloc✅ 触发, can_load=True                   ❌ 不触发
  load_specs[req_id]      LoadSpec(can_load=True, ...)             从未被设置
  build_connector_meta    ReqMeta(load_spec, save_spec)            from_request_tracker → None

RequestTracker:
  is_decode_phase         False                                    True (line 272)
  new_token_ids           完整 prompt tokens                       [1 token]

ReqMeta.from_request_tracker:
  skip_save               取决于 chunk 边界                        True (line 343)
  load_spec               LoadSpec 或 None                         None
  返回值                  ReqMeta(...)                             None (line 348)

Worker:
  start_load_kv           retrieve_layer() × N generator           空 for 循环
  wait_for_layer_load     next() × N 层 × generator               空列表循环
  save_kv_layer           store_layer() × N generator              无请求
  wait_for_save           推进 storer + unpin                      仅 unpin
```

---

## 特殊场景：`save_decode_cache=True`

当配置 `LMCACHE_SAVE_DECODE_CACHE=true` 时，decode 阶段**可以**保存 KV cache。

### 条件判断链（`vllm_v1_adapter.py:340-345`）

```python
skip_save = tracker.disagg_spec is None and (
    tracker.skip_save
    or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)  # ← 唯一剩余的 guard
    or (tracker.is_decode_phase and not save_decode_cache)                  # False, 不再阻止
    or request_skip
)
```

此时 `is_decode_phase and not save_decode_cache` 为 `False`，不再阻断。
但还有一个 chunk 边界 guard：

- `num_saved_tokens > 0`：prefill 阶段已保存过一部分 token
- `input_token_len < chunk_boundary`：当前总 token 数未达到下一个 chunk 边界

**所以 decode 保存是累积式的**——每积累 `lmcache_chunk_size`（默认 256）个新 decode token，
才会触发一次保存。此时 `save_kv_layer` 和 `wait_for_save` 会走完整的 layerwise store 流水线，
与 prefill 的 store 流程相同。

**注意**：即使 `save_decode_cache=True`，`start_load_kv` 和 `wait_for_layer_load`
在 decode 阶段仍然是 no-op——因为 decode 没有 `load_spec`，不会从缓存加载 KV
（vLLM 已在本地 GPU 上持有前面所有 token 的 KV cache）。

---

## 控制流总结

```
                         ┌─────────────────────────────┐
                         │  vLLM Scheduler             │
                         │                             │
   Prefill (new req) ──→ │  WAITING 队列               │
     num_computed=0       │  ├─ lookup() ✅             │
                          │  ├─ get_num_new_matched ✅  │
                          │  └─ update_state_after ✅   │
                          │                             │
   Decode (running) ───→ │  RUNNING 队列               │
     num_computed>0       │  └─ (无 connector 调用) ❌  │
                          └──────────┬──────────────────┘
                                     │
                                     ▼
                         ┌─────────────────────────────┐
                         │  build_connector_meta()      │
                         │                             │
   Prefill req ─────────→│  load_spec = LoadSpec(...)  │
                         │  save_spec = SaveSpec(...)   │
                         │  → ReqMeta 对象              │
                         │                             │
   Decode req ─────────→ │  load_spec = None            │
   (default config)      │  skip_save = True            │
                         │  → return None  (被排除)     │
                         │                             │
   Decode req ─────────→ │  load_spec = None            │
   (save_decode_cache)   │  skip_save 取决于 chunk 边界 │
                         │  → 可能生成 ReqMeta          │
                         └──────────┬──────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────────────┐
                         │  Worker: 4 个 connector 方法 │
                         │                             │
   有 ReqMeta 的请求 ──→ │  执行完整 layerwise 流水线   │
   无 ReqMeta 的请求 ──→ │  for 循环 continue (no-op)  │
                         └─────────────────────────────┘
```

---

## 关键源文件行号速查表

| 差异点 | 文件 | 行号 | 说明 |
|---|---|---|---|
| `is_decode_phase` 置位 | `vllm_v1_adapter.py` | 271-272 | `len(new_token_ids) == 1` 时设为 `True` |
| decode 跳过保存的 guard | `vllm_v1_adapter.py` | 343 | `is_decode_phase and not save_decode_cache` |
| `skip_save + load_spec=None` → 不生成元数据 | `vllm_v1_adapter.py` | 347-348 | `return None` |
| `save_decode_cache` 配置定义 | `config.py` | 102-106 | 默认 `False` |
| cached_reqs 的 `load_spec` | `vllm_v1_adapter.py` | 1658-1659 | `self.load_specs.pop()` 返回 `None` |
| decode token 切片 | `vllm_v1_adapter.py` | 1644-1651 | `num_new_tokens=1`，仅取最新 1 个 token |
| `start_load_kv` 的 per-request 跳过 | `vllm_v1_adapter.py` | 785 | `load_spec is None → continue` |
| `save_kv_layer` 的 per-request 跳过 | `vllm_v1_adapter.py` | 1011-1013 | `save_spec is None or not can_save → continue` |
| chunk 边界计算 | `vllm_v1_adapter.py` | 332-334 | `cdiv(saved+1, chunk_size) * chunk_size` |

---

## 一句话总结

> **Layerwise 是 prefill 优化**：通过 scheduler 侧的 `load_spec` / `save_spec` 元数据过滤，
> decode 阶段的 4 个 connector 方法全部退化为轻量 no-op。
> `LMCACHE_USE_LAYERWISE` 的流水线加速效果完全体现在 prefill 阶段的
> prefix KV 加载与新 KV 保存中。
