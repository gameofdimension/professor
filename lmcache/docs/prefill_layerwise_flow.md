# Prefill 阶段 Layerwise 完整流程（分支 v0.4.5-cu129）

> 以 `VLLMPagedMemLayerwiseGPUConnector`（非 blending 的标准 layerwise 路径）为例，
> vLLM 模型为 N 层 Transformer。
> 整个流程分为 **4 个阶段**，按 vLLM scheduler → worker → model forward 的调用顺序展开。

---

## 阶段 0：调度 — `lookup()` 确定 prefix 命中长度

在 prefill 调度阶段，scheduler 调用 `lookup()` 检查已缓存的 prefix token 数量。

**入口：** `cache_engine.py:1063` `lookup()` 方法

```
vLLM scheduler
  → LMCacheConnectorV1Dynamic.update_state()         # vllm 调度侧
    → lmcache_engine.lookup(tokens, pin=True)          # 查询缓存
```

### 核心逻辑（`cache_engine.py:1126-1150`）

由于 `use_layerwise=True`，走 layerwise 分支：

1. **Line 1118-1123**：调用 `token_database.process_tokens()` 将 token 序列按 chunk 大小切分，逐 chunk 迭代
2. **Line 1132**：对每个 chunk 的 key 调用 `key.split_layers(self.num_layers)`（定义于 `utils.py:449-464`），生成 N 个 `LayerCacheEngineKey`，每个带独立的 `layer_id`
3. **Line 1134-1138**：调用 `storage_manager.batched_contains(key_all_layers, search_range, pin=True)` 检查所有层的 key 是否都存在且在同一存储位置
4. **Line 1141**：只有 `hit_chunks == num_layers`（所有层都命中）且 `len(block_mapping) == 1`（同位置）才算命中
5. **Line 1148**：命中则 `res = end`，继续检查下一个 chunk；**任一 chunk 不全命中则立即 `return res`**（prefix 匹配，中断即止）

返回值 `res` = 已缓存的 prefix token 数量，告诉 vLLM 需要从第 `res` 个 token 开始 prefill。

---

## 阶段 1：加载开始 — `start_load_kv()`

vLLM worker 在执行 model forward 前调用此方法，启动逐层加载的 generator。

**入口：** `vllm_v1_adapter.py:738`

### 执行流程

#### 1.1 遍历请求，构建 load 参数

- **Line 770-773**：找到最后一个可加载请求的 index `last_idx`（用于最后一个请求的 `sync=True`，确保 CUDA stream 同步）
- **Line 788-799**：构建 `token_mask`，将 vLLM 已缓存的 token 标记为 `False`（不需要加载），其余标记为 `True`

#### 1.2 创建 retrieve_layer generator

- **Line 802**：`if self.use_layerwise:` 进入 layerwise 分支
- **Line 803-806**：最后一个请求设 `sync=True`（等待 load stream 完成），其余 `sync=False`（不阻塞，流水线化）
- **Line 818-824**：调用 `self.lmcache_engine.retrieve_layer(tokens, mask, kvcaches, slot_mapping, sync)` 创建 generator

**进入 `retrieve_layer()`（`cache_engine.py:908`）：**

- **Line 967-996**：遍历 `token_database.process_tokens()` 的每个 chunk：
  - **Line 974**：`key.split_layers(self.num_layers)` 拆成逐层 key
  - **Line 977-978**：`storage_manager.contains(keys_multi_layer[0])` 只检查第一层是否存在（快速判断）
  - **Line 984-988**：所有 chunk 必须在同一存储位置
  - **Line 992-996**：收集 `starts`、`ends`、`keys`，设置 `ret_mask`
- **Line 999-1000**：将 keys 转置为 **layer-major** 格式：`keys_layer_major[layer_id][chunk_idx]`
- **Line 1002-1005**：调用 `storage_manager.layerwise_batched_get(keys_layer_major)` 获得异步 get generator（每层返回一个 `Future`）
- **Line 1009-1010**：调用 `gpu_connector.batched_to_gpu(starts, ends)` 获得传输 generator，`next()` 初始化它

**进入 `batched_to_gpu()`（`gpu_connectors.py:1174`）：**

- **Line 1194-1208**：初始化 kvcaches 指针、分配 GPU 临时 buffer、切分 `slot_mapping`
- **Line 1236**：进入 `for layer_id in range(num_layers)` 循环，**`memory_objs_layer = yield`**——等待调用方 `send()` 传入该层的 memory objects
- **此时 generator 暂停在第 0 层的 `yield` 处**

#### 1.3 推进 generator 两次

回到 `start_load_kv()`：

- **Line 827**：`next(layerwise_retriever)` → 推进 `retrieve_layer` 的第 1 个 yield
  - `retrieve_layer` 内部执行 `task = next(get_generator)` 获取 layer 0 的 `Future`（`cache_engine.py:1014`）
  - **Line 1018-1021**：`yield torch.sum(ret_mask)` → 返回命中 token 数（返回给 `start_load_kv` 但未被使用）
- **Line 828**：`next(layerwise_retriever)` → 推进第 2 个 yield
  - **Line 1025**：`task.result()` 阻塞等待 layer 0 的内存对象从存储后端到达
  - **Line 1026**：`mem_obj_consumer.send(mem_objs_layer)` 将 layer 0 的内存对象送入 `batched_to_gpu` generator
  - **Line 1023**：`yield None`
  - `batched_to_gpu` 内部（`gpu_connectors.py:1244-1281`）：在 `load_stream` 上将 layer 0 的 CPU 内存对象 → GPU buffer → paged kvcaches，然后 `yield` 等待下一层
  - 然后 `retrieve_layer` 继续：`task = next(get_generator)` 获取 layer 1 的 `Future`，进入下一次 `yield None`

- **Line 829**：`self.layerwise_retrievers.append(layerwise_retriever)` — generator 暂停在「layer 0 已传输到 GPU，layer 1 正在从存储后端异步加载到 CPU 内存」的状态

**此时状态：** layer 0 的 KV 已在 GPU 上，layer 1 正在从存储后端异步加载到 CPU 内存。

---

## 阶段 2：逐层计算 — 模型 forward 中的 `wait_for_layer_load()`

vLLM 模型 forward pass 中，每一层 Transformer 执行前调用此方法。

**入口：** `vllm_v1_adapter.py:944`

```
vLLM model forward（循环 N 层）
  for layer_id in range(N):
    → connector.wait_for_layer_load(layer_name)     # 等该层 KV 就绪
    → model.forward(layer_id)                        # 执行 attention + FFN
    → connector.save_kv_layer(layer_name, kv_layer)  # 保存该层新计算的 KV
```

### 2.1 `wait_for_layer_load(layer_name)`（`vllm_v1_adapter.py:953-968`）

- **Line 957-958**：对所有 `layerwise_retriever` generator 调用 `next()`

**每次 `next()` 推进 `retrieve_layer` generator 一步（回到 `cache_engine.py:1013-1027`）：**

```
第 i 次 next() (i = 1, 2, ..., N-1):
├── cache_engine.py:1014  task = next(get_generator)        # 获取 layer i+1 的 Future
├── cache_engine.py:1021  yield None                        # 返回控制权
│   (下一次 next 继续 ↓)
├── cache_engine.py:1025  mem_objs_layer = task.result()    # 等待 layer i 的内存到达
├── cache_engine.py:1026  mem_obj_consumer.send(mem_objs)   # 送入 batched_to_gpu
│   └── gpu_connectors.py:1237  memory_objs_layer = yield   # 接收内存对象
│       gpu_connectors.py:1244  with torch.cuda.stream(load_stream):
│       gpu_connectors.py:1259-1261  CPU memobj → GPU buffer (non_blocking)
│       gpu_connectors.py:1274-1281  GPU buffer → paged kvcaches (scatter write)
│       gpu_connectors.py:1282  yield                       # 等待下一层
```

**流水线时序图（以 4 层为例）：**

```
时间轴 →

current_stream:  │ wait(0) │ L0 attn+FFN │ wait(1) │ L1 attn+FFN │ wait(2) │ L2 attn+FFN │ wait(3) │ L3 attn+FFN │
load_stream:     │ load L0 │   load L1   │   load L2   │   load L3   │
store_stream:    │         │  save L0    │  save L1    │  save L2    │  save L3  │

wait(i) = current_stream.wait_stream(load_stream) 确保第 i 层加载完毕
```

- **Line 1238-1239**（`gpu_connectors.py:1238`）：当 `sync=True`（最后一个请求），执行 `current_stream.wait_stream(self.load_stream)` 确保加载完成后再继续

- **Line 960-963**（`vllm_v1_adapter.py:960`）：最后一层时，generator 的最后一个 yield 返回 `ret_mask`（`cache_engine.py:1061`），记录成功加载的 token 数

- **Line 966**：`self.current_layer += 1`

### 2.2 `save_kv_layer(layer_name, kv_layer)`（`vllm_v1_adapter.py:971-1070`）

每层计算完成后调用。将新计算的 KV 写入缓存。

- **Line 990-991**：`if not self.use_layerwise: return` — 非 layerwise 模式不在此处保存
- **Line 1016-1066**：对每个请求，**首次调用时**创建 `store_layer()` generator：
  - **Line 1037-1041**：计算 `skip_leading_tokens`（已缓存的前缀不重复存储），按 chunk 大小对齐
  - **Line 1057-1065**：调用 `self.lmcache_engine.store_layer(token_ids, mask, kvcaches, slot_mapping, offset, sync)` 创建 generator
  - **Line 1066**：缓存在 `self._layerwise_save_storers[req_id]` 中

**进入 `store_layer()`（`cache_engine.py:569`）：**

- **Line 645-705**：遍历 `process_tokens()` 的每个 chunk：
  - **Line 650**：`key.split_layers(self.num_layers)` 拆成逐层 key
  - **Line 652-655**：如果第一层已存在则 `continue`（去重）
  - **Line 659-667**：`storage_manager.batched_allocate()` 为该 chunk 的**所有层**预分配 CPU 内存对象
  - **Line 709**：转置为 layer-major 格式
- **Line 720-724**：创建 `gpu_connector.batched_from_gpu(memory_objs, starts, ends)` generator，`next()` 初始化

**进入 `batched_from_gpu()`（`gpu_connectors.py:1296`）：**

- **Line 1368**：进入 `for layer_id in range(num_layers)` 循环
- **Line 1371-1372**：`with torch.cuda.stream(self.store_stream): store_stream.wait_stream(current_stream)` — 等模型 forward 当前层完毕
- **Line 1374-1399**：paged kvcaches → GPU buffer → CPU memory objects（D2H）
- **Line 1404**：`yield` — 等待调用方推进

回到 `store_layer()`：

- **Line 726-731**（`cache_engine.py:726`）：
  ```python
  for layer_id in range(self.num_layers):
      yield                                          # 等待调用方 next()
      next(mem_obj_generator)                        # 推进 GPU→CPU 传输
      self.storage_manager.batched_put(              # 将上一层的 CPU 内存写入存储后端
          keys[layer_id], memory_objs[layer_id], location=self.store_location
      )
  ```

**在 `save_kv_layer()` 中：**

- **Line 1070**（`vllm_v1_adapter.py:1070`）：`next(layerwise_storer)` — 每次 `save_kv_layer` 调用推进 generator 一步

**每次推进的效果：**

```
第 0 次 next(): yield → 分配所有层 CPU 内存, 启动 GPU→CPU 传输 layer 0
第 1 次 next(): yield → 完成 layer 0 传输, batched_put(layer 0), 启动 layer 1 传输
...
第 N-1 次 next(): yield → 完成 layer N-2 传输, batched_put(layer N-2), 启动 layer N-1 传输
```

---

## 阶段 3：收尾 — `wait_for_save()`

模型 forward 全部层执行完毕后调用。

**入口：** `vllm_v1_adapter.py:1073`

- **Line 1091**：`if self.use_layerwise:` 进入 layerwise 分支
- **Line 1092-1099**：对所有请求：
  - **Line 1093-1097**：`pop` 出 `layerwise_storer` generator，调用最后一次 `next()`
    - 这推进 `store_layer` 的最后一次迭代：完成最后一层的 GPU→CPU 传输 + `batched_put(最后一层)`
    - 然后 `store_layer` 的 `cache_engine.py:747-751` 处理无 chunk 可存的情况并 `yield` 结束
  - **Line 1099**：`lookup_unpin(req_id)` — 释放阶段 0 中 `lookup(pin=True)` 锁定的缓存引用

---

## 完整时序图（3 层模型，1 个请求，prefix 命中 2 chunks）

```
                      Scheduler Worker                     LMCache Engine              StorageManager         GPU Connector
                          │                                    │                            │                       │
Phase 0: lookup           │──lookup(tokens, pin=True)──────→│                            │                       │
                          │                                    │──split_layers──────────→│                       │
                          │                                    │──batched_contains────────→│ (check all layers)  │
                          │←──res=512 (cached tokens)────────│                            │                       │
                          │                                    │                            │                       │
Phase 1: start_load_kv    │──start_load_kv()───────────────→│                            │                       │
                          │                                    │──retrieve_layer()────→   │                       │
                          │                                    │  process_tokens           │                       │
                          │                                    │  split_layers             │                       │
                          │                                    │  layerwise_batched_get──→│ (async get L0,L1,L2) │
                          │                                    │──batched_to_gpu()────────────────────────────→│
                          │                                    │  next() #1: yield L0 Future                      │
                          │  next() #1 ─────────────────────→│  next() #2: L0.result()→send to connector       │
                          │                                    │                          │←──get L0 from store──│
                          │                                    │                            │   load L0→GPU        │
                          │                                    │  (L1 async loading started)│                      │
                          │  append retriever                  │                            │                       │
                          │                                    │                            │                       │
Phase 2: model forward    │                                    │                            │                       │
  Layer 0:                │                                    │                            │                       │
   wait_for_layer_load(0) │──next()────────────────────────→│  L1.result()→send to conn │←──get L1 from store──│
                          │  (L0 already on GPU)              │  load L1→GPU (async)       │   load L1→GPU        │
                          │                                    │  get L2 Future             │                       │
   model.forward(L0)      │  ═══ L0 attention + FFN ═══      │                            │                       │
   save_kv_layer(L0)      │──next(storer)──────────────────→│  GPU→CPU L0 (store_stream) │                       │
                          │                                    │                            │                       │
  Layer 1:                │                                    │                            │                       │
   wait_for_layer_load(1) │──next()────────────────────────→│  L2.result()→send to conn │←──get L2 from store──│
                          │  (L1 already on GPU)              │  load L2→GPU (async)       │   load L2→GPU        │
   model.forward(L1)      │  ═══ L1 attention + FFN ═══      │                            │                       │
   save_kv_layer(L1)      │──next(storer)──────────────────→│  put(L0) to store─────────→│                       │
                          │                                    │  GPU→CPU L1 (store_stream) │                       │
                          │                                    │                            │                       │
  Layer 2:                │                                    │                            │                       │
   wait_for_layer_load(2) │──next()────────────────────────→│  (last layer)              │                       │
                          │  (L2 already on GPU)              │  ret_mask returned         │                       │
   model.forward(L2)      │  ═══ L2 attention + FFN ═══      │                            │                       │
   save_kv_layer(L2)      │──next(storer)──────────────────→│  put(L1) to store─────────→│                       │
                          │                                    │  GPU→CPU L2 (store_stream) │                       │
                          │                                    │                            │                       │
Phase 3: wait_for_save    │──wait_for_save()───────────────→│                            │                       │
                          │──next(storer) final────────────→│  put(L2) to store─────────→│                       │
                          │                                    │  lookup_unpin(req_id)─────→│                       │
                          │                                    │                            │                       │
```

---

## 关键源文件行号速查表

| 步骤 | 文件 | 行号 | 说明 |
|---|---|---|---|
| 配置读取 | `cache_engine.py` | 159 | `self.use_layerwise = config.use_layerwise` |
| 连接器选择 | `gpu_connector/__init__.py` | 96-97 | `VLLMPagedMemLayerwiseGPUConnector` |
| **Phase 0: lookup** | `cache_engine.py` | 1126-1150 | layerwise 分支，逐 chunk 检查所有层 |
| key 拆分 | `utils.py` | 449-464 | `split_layers()` 生成逐层 key |
| **Phase 1: start_load_kv** | `vllm_v1_adapter.py` | 802-829 | 创建 retriever generator，两次 `next()` |
| retrieve_layer 创建 | `cache_engine.py` | 908-1061 | generator，N+2 次 yield |
| 异步批量 get | `storage_manager.py` | 517-542 | `layerwise_batched_get()`，每层一个 Future |
| GPU 传输 (load) | `gpu_connectors.py` | 1174-1293 | `batched_to_gpu()`，N+2 次 yield |
| load_stream | `gpu_connectors.py` | 1088 | `self.load_stream = torch.cuda.Stream()` |
| **Phase 2: wait_for_layer_load** | `vllm_v1_adapter.py` | 944-968 | 每层一次 `next()` |
| **Phase 2: save_kv_layer** | `vllm_v1_adapter.py` | 971-1070 | 创建/推进 storer generator |
| store_layer 创建 | `cache_engine.py` | 569-751 | generator，N+1 次 yield |
| GPU 传输 (store) | `gpu_connectors.py` | 1296-1413 | `batched_from_gpu()`，N+1 次 yield |
| store_stream | `gpu_connectors.py` | 1089 | `self.store_stream = torch.cuda.Stream()` |
| CPU→存储后端写入 | `cache_engine.py` | 729-731 | `storage_manager.batched_put()` |
| **Phase 3: wait_for_save** | `vllm_v1_adapter.py` | 1091-1099 | 最后一次 `next()` + unpin |
| 三条 CUDA stream | `gpu_connectors.py` | 1088-1089 | `load_stream` / `store_stream` + `current_stream` |

---

## 三条 CUDA Stream 协作模型

| Stream | 用途 | 操作 |
|---|---|---|
| `current_stream` | 模型 forward 计算 | attention + FFN |
| `load_stream` | KV 加载（retrieve） | CPU memobj → GPU buffer → paged kvcaches |
| `store_stream` | KV 保存（store） | paged kvcaches → GPU buffer → CPU memobj |

同步点：
- `wait_for_layer_load` 中：`current_stream.wait_stream(load_stream)` 确保该层 KV 加载完毕后再计算
- `save_kv_layer` 中：`store_stream.wait_stream(current_stream)` 确保该层计算完毕后再拷贝 KV
- 只有 `sync=True`（最后一个请求）时才执行 `current_stream` 上的同步等待，其余请求完全异步
