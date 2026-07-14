<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 调查文档：Item-constrained Generation（物品约束生成）

> 调查对象：`examples/sid-gr-inference`（SID-GR Inference）项目
> README 相关条目：第 2 节 Approach（`item-constrained decode`）、第 3 节 Design Highlights、第 10 节 Roadmap
> 调查日期：2026-07-14

---

## 0. TL;DR

Item-constrained generation 是 SID-GR（Semantic-ID Generative Recommender）推理路径中的 **trie 约束解码（trie-based constrained decoding）**：用一棵覆盖物品目录里所有合法 semantic-ID 序列的前缀树，在每个 decode 步把非法 token 的 logit 置为 `-inf`，再做 topK，从而**保证每条 beam 结果最终都能映射回一个真实存在的目录物品**。

- **核心模块**：`src/gr_inference/gr_runtime/item_constraints.py`
- **mask 作用点**：`src/gr_inference/gr_runtime/batched_beam_search.py`（constrained topK）
- **serving 接入**：`src/gr_inference/gr_serving/{api,engine,http,request}.py`
- **启动配置**：`scripts/serve_qwen3_gr_http.sh` + `tools/serve_qwen3_gr_http.py`
- **单测示例**：`tests/test_item_constraints.py`

---

## 1. 背景：为什么需要约束

SID-GR 的核心工作流（README 第 1 节）：

```
离线聚类
-> 每个 real item ID 映射成一个多级 cluster ID tuple（semantic ID）
-> 推理时自回归生成一段短的 semantic ID 序列
-> 把生成的 semantic ID tuple 映射回 real item ID
```

该工作流的特点是 **long context + short decode + large beam**（cluster 深度通常只有 3 或 5）。

**痛点**：模型逐 token 自回归生成，beam search 会生成出大量"看起来合理、但目录中并不存在"的 semantic-ID tuple（即"幽灵 / 非法 item"）。如果不加约束：

1. 生成的结果可能映射不到任何真实 item，需要事后过滤，浪费 beam 预算；
2. beam 可能走进永远无法完成（非 terminal）的死前缀；
3. "映射回 item" 这一步变成概率性、需要兜底逻辑。

Item-constrained generation 用 trie 把"合法 token 序列"显式编码进解码过程，从根上消除这些问题。

---

## 2. 是什么：机制与核心抽象

### 2.1 机制

1. 离线把目录里每个 item 的 semantic-ID 序列插入一棵 `TokenTrie`。
2. 推理时，对每条 beam 当前已生成的前缀 `prefix`，trie 的 `allowed_next(prefix)` 返回所有合法的下一个 token。
3. 据此构造一个布尔 mask `[beam, vocab]`，合法 token 为 `True`、非法为 `False`。
4. 在做 topK 前，对 logits 执行 `logits.masked_fill(~mask, -inf)`（**constrained topK**），非法 token 永远不会被选中。
5. beam 走到 trie 的 terminal 节点即表示生成了一个完整、合法的 semantic-ID，可直接经 `resolve_item_ids` 取回真实 item id。

> 对 terminal item，provider 还允许在序列末尾放行 EOS token（`allow_eos_for_terminal`），用于固定长度生成语义下的"结束"标记。

### 2.2 核心类（`item_constraints.py`）

| 类 / 函数 | 位置 | 职责 |
| --- | --- | --- |
| `TokenTrieNode` | `item_constraints.py:20` | trie 节点：`children`、`terminal`、`item_ids` |
| `TokenTrie` | `item_constraints.py:424` | 目录所有 semantic-ID 序列的前缀树；`from_sequences` / `from_items` / `insert` / `allowed_next` / `is_terminal` / `item_ids` |
| `SemanticItem` | `item_constraints.py:27` | 一个目录物品：`item_id` + `token_ids` + `metadata`（frozen dataclass） |
| `SemanticItemCatalog` | `item_constraints.py:35` | 经校验的目录；`from_records` / `from_jsonl` 加载，校验重复 item_id、重复 token path、token 范围 `< vocab_size` |
| `TrieItemMaskProvider` | `item_constraints.py:490` | 由 trie + 当前 BeamPath 生成 torch mask；`initial_mask` / `step_mask` / `allowed_next` / `resolve_item_ids` / `beam_item_results` |
| `TrieItemMaskProviderStore` | `item_constraints.py:165` | 原子化持有器，支持 `reload_jsonl` / `rollback` / 版本号 / reload 历史 |

`TrieItemMaskProvider` 关键方法：

- `initial_mask(logits)`（`:514`）：prefill 后用，返回 `[vocab]` 的初始合法 token mask（根节点 `allowed_next(())`）。
- `step_mask(generation, logits)`（`:524`）：每个 decode 步用，根据每条 beam 的 `beam_path.token_trace(beam)` 前缀生成 `[W, vocab]` mask。**当前直接实现要求 `batch_size==1`**（`:531-532`），serving engine 通过逐请求切片来规避（见第 4 节）。
- `beam_item_results(beam_path, beam_width)`（`:564`）：把 beam 结果映射回 `item_id` / `semantic_token_ids` / `is_complete`。

### 2.3 mask 如何作用于 topK（`batched_beam_search.py`）

- `select_initial_topk_batched(item_mask=...)`（`:63`）：mask 经广播后 `masked_fill(~mask, -inf)`（`:99`），再 `torch.topk`；若合法 token 少于 `beam_width` 直接报错（`:104-105`）。
- `select_next_topk_batched(item_mask=...)`（`:119`）：同理，对 `[B, W_prev, V]` 的 decode logits 应用 mask（`:159`）。
- `batched_item_mask_limited_beam_width(beam_width, item_mask)`（`:217`）：把 batch 内 `beam_width` **夹紧**到各行合法候选数的最小值，避免"合法 token 不够"导致报错。

---

## 3. 为什么：价值

1. **输出必然可落地** —— 每条完成的 beam 都能经 trie terminal 节点直接 `resolve_item_ids` 取回真实 item id，不存在"生成了但推荐不出物品"。
2. **质量更高** —— beam 预算不再浪费在死前缀上，所有 beam 沿合法路径推进。
3. **目录可热更** —— 推荐系统的物品目录动态变化，`TrieItemMaskProviderStore` 提供**原子 swap + 版本号 + 回滚上一版本 + reload 历史**，无需重启服务。
4. **与 SID-GR 原生抽象一致** —— 约束以请求级 `item_mask_provider` 挂在 `BeamPath` 上，符合 README 强调的"保留 SID-GR 原生抽象（`ContextKV` / `BeamKV` / `BeamPath` / item-constrained decode / request×active-beam batching 直接由 runtime 表达）"。
5. **可观测** —— catalog 版本、reload 历史、每请求 `has_item_constraints` 都暴露在 status / metrics 中。

---

## 4. 架构与数据流

### 4.1 serving 端请求级接入（`api.py`）

启动时若配置了 catalog，会构造一个全局 `TrieItemMaskProviderStore` 挂在 facade 上（`api.py:24`）。每个进来的请求在 `_prepare_request`（`api.py:188-197`）里被**自动注入** store 的当前 snapshot：

```python
def _prepare_request(self, request):
    if request.item_mask_provider is not None or self.item_mask_provider_store is None:
        return request                      # 请求自带 provider 或未配置目录 -> 不动
    return replace(request,
                   item_mask_provider=self.item_mask_provider_store.snapshot())
```

请求字段定义在 `request.py:27`（`item_mask_provider: Any | None = None`）。请求状态里用 `has_item_constraints`（`api.py:238`）标记是否启用了约束。

> snapshot 模型带来的语义：**进行中的请求继续用它捕获时的旧版本 provider，新请求拿到新版本**，reload 期间不会撕裂。

### 4.2 mask 在 engine 里的拼装（`engine.py`）

engine 对一个连续批次的多个请求拼出 batched mask：

- `_batched_initial_item_mask(requests, logits)`（`engine.py:510`）：逐请求调用 `provider.initial_mask(scores[b:b+1])`，stack 成 `[B, V]`；无 provider 的请求行填全 `True`。
- `_batched_step_item_mask(requests, batched_beam_path, logits)`（`engine.py:539`）：逐请求调用 `provider.step_mask(...)`（`engine.py:562`），传入 `logits[b:b+1]`（因此 `step_mask` 的 `batch_size==1` 断言恒成立），stack 成 `[B, W, V]`。

随后在 beam 选择处传入：

- prefill 后初始选择（`engine.py:213-223`）：`select_initial_topk_batched(..., item_mask=initial_item_mask)`，并用 `batched_item_mask_limited_beam_width` 夹紧宽度。
- 每个 decode 步（`engine.py:286-299`）：`select_next_topk_batched(..., item_mask=item_mask)`，同样夹紧宽度。

### 4.3 目录热更与版本管理（`item_constraints.py:165-356`）

`TrieItemMaskProviderStore` 用 `RLock` 保护，提供：

- `snapshot()`（`:234`）：返回当前 provider（供请求注入）。
- `swap(provider, metadata, operation)`（`:238`）：原子替换，版本号自增，旧 provider/版本/metadata 存入 `_previous_*`，记一条 reload 历史。
- `reload_jsonl(path, ...)`（`:264`）：重新加载 JSONL 构造 provider 后 swap；失败时记录失败事件并抛出。
- `rollback()`（`:310`）：把 `_previous_*` 换回当前，版本号再次自增（回滚也是一次"前进"），再次 swap 保留可回滚性。
- `status()`（`:336`）：`version` / `previous_version` / `reload_history` / `last_reload` / 目录 metadata。

reload 历史上限默认 16 条（`max_reload_history`）。

---

## 5. 怎么用

### 5.1 目录文件格式（JSONL，每行一条）

```json
{"item_id": "item-a", "token_ids": [1, 10], "metadata": {"score": 0.9}}
{"item_id": "item-b", "token_ids": [2, 11]}
```

- 字段名可通过 `item_id_field` / `token_ids_field` / `metadata_field` 配置（默认即 `item_id` / `token_ids` / `metadata`）。
- `vocab_size` 用于校验每个 token 在词表范围内；超出会报 `exceeds vocab_size`。
- 默认禁止重复 `item_id` 与重复 token path，可经 `allow_duplicate_item_ids` / `allow_duplicate_token_paths` 放开。

### 5.2 方式 A：HTTP serving（生产路径）

启动时带目录，环境变量（`scripts/serve_qwen3_gr_http.sh:79-95`）：

```bash
GR_CATALOG_JSONL=path/to/catalog.jsonl \
GR_CATALOG_VOCAB_SIZE=151936 \        # 缺省取模型 vocab_size
GR_CATALOG_EOS_TOKEN_ID=151643 \
GR_ALLOW_CATALOG_RELOAD=1 \           # 启用 /catalog/reload 与 /catalog/rollback
scripts/serve_qwen3_gr_http.sh
```

等价 CLI 参数（`tools/serve_qwen3_gr_http.py:662-674, 724`）：

```
--catalog-jsonl PATH
--catalog-vocab-size N            (缺省 = 模型 vocab_size)
--catalog-eos-token-id N
--catalog-item-id-field item_id
--catalog-token-ids-field token_ids
--catalog-metadata-field metadata
--catalog-allow-eos-for-terminal  (默认 True)
--catalog-allow-duplicate-item-ids
--catalog-allow-duplicate-token-paths
--allow-catalog-reload            (启用 reload/rollback 端点)
```

**配好目录后无需修改请求**：`/generate`、`/submit`、`/submit_many` 请求会被 `_prepare_request` 自动注入全局 snapshot，constrained topK 自动生效。可用 `/requests/<id>` 返回的 `has_item_constraints: true` 确认。

目录管理 HTTP 端点（`http.py:199 / 279 / 304`；路由清单 `http.py:983-990`）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/catalog/status` | `version` / `item_count` / `previous_version` / `reload_history` / `last_reload` |
| POST | `/catalog/reload` | body：`{path, vocab_size, eos_token_id, allow_eos_for_terminal, item_id_field, token_ids_field, metadata_field, allow_duplicate_item_ids, allow_duplicate_token_paths}` → 返回新 `version` 与 catalog status（需 `--allow-catalog-reload`） |
| POST | `/catalog/rollback` | 回滚到上一 version，返回新 version 与 status（需 `--allow-catalog-reload`） |

### 5.3 方式 B：离线 / 编程方式

```python
from gr_inference.gr_runtime import (
    SemanticItemCatalog,
    select_initial_topk_batched,
    select_next_topk_batched,
)

# 1) 加载目录并构造 mask provider
catalog  = SemanticItemCatalog.from_jsonl(
    "items.jsonl", vocab_size=151936, eos_token_id=151643,
)
provider = catalog.provider(vocab_size=151936, eos_token_id=151643)

# 2) prefill 后用 initial mask 选初始 beam
initial_mask = provider.initial_mask(prefill_logits)                 # [V]
sel0 = select_initial_topk_batched(
    prefill_logits, beam_width=256, item_mask=initial_mask,
)

# 3) 每个 decode 步用 step mask（基于每条 beam 的前缀）
step_mask = provider.step_mask(generation, decode_logits)            # [W, V]
sel_n = select_next_topk_batched(
    decode_logits, previous_scores=..., beam_width=256, item_mask=step_mask,
)

# 4) 映射回真实 item id
results = provider.beam_item_results(beam_path, beam_width=256)
# -> [{"rank", "token_ids", "semantic_token_ids", "is_complete", "item_ids", "item_id"}, ...]
```

运行时热更（编程方式）：

```python
store = TrieItemMaskProviderStore.from_jsonl("v1.jsonl",
                                             vocab_size=151936, eos_token_id=151643)
new_version = store.reload_jsonl("v2.jsonl", vocab_size=151936, eos_token_id=151643)
store.rollback()   # 回到 v1
```

### 5.4 单测作为可直接运行的示例

`tests/test_item_constraints.py` 覆盖：

- trie 的 `allowed_next` / `is_terminal` / `item_ids` 行为；
- provider 的 `initial_mask` / `step_mask` 与 "EOS-for-terminal" 放行；
- catalog 的 JSONL 加载、metadata 解析、重复 path / 越界 token 校验；
- store 的 `reload_jsonl` 版本切换与旧 snapshot 仍可解析旧 item。

---

## 6. 局限与路线图

代码与 README 第 10 节 Roadmap 明确标注的现状（alpha）：

1. **`step_mask` 直接实现要求 `batch_size==1`**（`item_constraints.py:531-532`）。serving engine 通过逐请求切片规避（`engine.py:562`），但离线直接批调用需注意。
2. **未在真实大目录上验证**：Roadmap 待办为"用真实大目录测试；补 item 级正确性、非法 token 校验、constrained topK 优化、serving 语义回归"。
3. **constrained topK 性能**：当前是 `masked_fill(-inf) + topk` 的通用实现，大词表 / 大 beam 下有优化空间（Roadmap 提及 constrained topK 优化）。
4. **beam 选择仍在 graph 外**：README 第 10 节"Beam selection graph"待办提到，需先把 item mask、special-token 抑制、输出裁剪等都安全化后，才能把 `log_softmax + topK + beam selection` 整体并入 CUDA graph。当前带 item mask 的 decode 路径会走 eager（与"fixed-shape decode CUDA graph"热路径存在交互）。

---

## 7. 相关文件索引

| 文件 | 关键内容 |
| --- | --- |
| `src/gr_inference/gr_runtime/item_constraints.py` | trie / catalog / mask provider / store 全部核心实现 |
| `src/gr_inference/gr_runtime/batched_beam_search.py` | constrained topK（`select_initial_topk_batched` / `select_next_topk_batched` / `batched_item_mask_limited_beam_width`） |
| `src/gr_inference/gr_serving/api.py` | facade：catalog status / reload / rollback、`_prepare_request` 自动注入 |
| `src/gr_inference/gr_serving/engine.py` | 连续批处理下 batched mask 的拼装与接入 |
| `src/gr_inference/gr_serving/http.py` | `/catalog/{status,reload,rollback}` 端点 |
| `src/gr_inference/gr_serving/request.py` | 请求字段 `item_mask_provider` |
| `tools/serve_qwen3_gr_http.py` | 启动 CLI：`--catalog-*` / `--allow-catalog-reload` |
| `scripts/serve_qwen3_gr_http.sh` | 启动环境变量：`GR_CATALOG_*` / `GR_ALLOW_CATALOG_RELOAD` |
| `tests/test_item_constraints.py` | 行为与用法示例单测 |
