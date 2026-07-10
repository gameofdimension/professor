# SID-GR Inference 框架调查报告

> 研究对象：`examples/sid-gr-inference/`（自包含的 GR 推理 + serving 框架）
> 外部依赖：`corelib/gr_decode_atten/`（被该框架以可插拔后端方式调用的 beam decode attention kernel）
> 调查方式：以读代码为主，文档仅作参照；下文结论均附 `file:line`。

---

## 目录

1. [项目定位](#1-项目定位)
2. [业务背景：为什么需要专门做这个](#2-业务背景为什么需要专门做这个)
3. [包结构与公共 API](#3-包结构与公共-api)
4. [核心抽象：KV 三件套 + GRGenerationState](#4-核心抽象kv-三件套--grgenerationstate)
5. [执行分层：离线库 / 同步 serving / 连续拼批 serving](#5-执行分层离线库--同步-serving--连续拼批-serving)
6. [kernel 作为可插拔后端：框架↔kernel 契约](#6-kernel-作为可插拔后端框架kernel-契约)
7. [serving 内部架构](#7-serving-内部架构)
8. [服务启动详细过程](#8-服务启动详细过程)
9. [请求处理全流程](#9-请求处理全流程)
10. [外部 kernel：gr_decode_atten 算法简述](#10-外部-kernelgr_decode_atten-算法简述)
11. [运转方式：环境变量 + shell + 容器](#11-运转方式环境变量--shell--容器)
12. [文档说法 vs 代码实证](#12-文档说法-vs-代码实证)
13. [现状、局限与 TODO](#13-现状局限与-todo)
14. [一句话总结](#14-一句话总结)

---

## 1. 项目定位

本目录是一个**自包含的 Python 推理框架**，包名 `gr_inference`（外加一个 JIT 扩展 `gr_inference_trtllm_kernels`）。定位是 **GR（生成式推荐）专用推理 + serving 框架**，对标 vLLM/SGLang/TensorRT-LLM，但**只为「长 context + 短 decode + 超大 beam」这一种负载做特化**。它不是某个大框架的插件，而是自己定义了全套抽象（KV、beam、调度、池、HTTP），只在「attention kernel」这一个点上通过可插拔后端去复用外部成果。

一句代码证据——核心 decode attention 的后端是注入式 callable：

```python
GRDecodeAttention(backend=lambda inputs: inputs.q)          # 测试/stub：恒等
GRDecodeAttention(backend=ExistingGRDecodeAttentionBackend())  # 生产：调外部 kernel
```

- stub：`examples/run_tiny_gr.py:81`
- 生产：`tools/serve_qwen3_gr_http.py` 经 `make_decode_backend` 构造

**框架与 kernel 是解耦的，这是理解整个目录的钥匙。**

---

## 2. 业务背景：为什么需要专门做这个

推荐/搜索/广告里的 **SID-GR（基于 Semantic ID 的生成式推荐）** 工作流：

```
离线聚类 → 每个 item 映射成多层 cluster ID 元组
→ 推理时自回归生成一串短 semantic ID → 映射回真实 item
```

这导致推理负载形状非常特殊，与聊天式 LLM 截然不同：

| 维度 | 通用 LLM serving | SID-GR 推理 |
|---|---|---|
| context | 较短、多请求 | **超长**（用户历史，几千 token） |
| decode | 长、持续生成 | **极短**（cluster 深度 3~5 步） |
| beam width | 小（<10，多样性） | **巨大**（128/256，召回海量候选） |
| batch | 多请求动态拼批 | 请求少、每请求 beam 巨多 |

通用框架（vLLM/SGLang/TRT-LLM）会把 `batch × beam`（如 4×256=1024）**拍平成 1024 条独立 decode 行**，每行都对「长 context + 历史」跑一遍 paged decode attention——这会把同一条长 context **重复读 beam_width 次**，正是性能黑洞。

本框架的立身之本：**为「长 context + 短 decode + 大 beam」专门设计 KV 结构与 decode attention**，把 SGLang beam-search 分支在 H100/Qwen3-1.7B 上跑出约 2.2× 离线加速、1.85× 在线吞吐（README 数据，机制见 §10/§12）。

---

## 3. 包结构与公共 API

框架对外导出约 60 个符号（`src/gr_inference/__init__.py`），分 6 个子包：

| 子包 | 职责 | 代表导出 |
|---|---|---|
| `gr_models/qwen3/` | 模型（Qwen 族） | `Qwen3GRConfig` `Qwen3GRModel` `materialize_qwen3_checkpoint` |
| `gr_kv/` | **KV 与 beam 血缘抽象**（框架灵魂） | `ContextKV` `BeamKV` `BeamPath` `TensorSpec` |
| `gr_kernels/` | kernel 封装 + 后端选择（**故意不 import torch**） | `GRDecodeAttention` `PrefillAttention` `TorchSDPAPrefillBackend` |
| `gr_runtime/` | beam search 运行时、logits、item 约束 | `GRDecodeEngine` `GRGenerationState` `FixedBeamDecodeLoop` `TokenTrie` `TrieItemMaskProvider` |
| `gr_scheduler/` | beam 宽度策略 | `FixedBeamPolicy` `ScheduledBeamPolicy` `ScoreMarginBeamPolicy` |
| `gr_serving/` | 连续拼批、内存池、CUDA graph、HTTP | `GRContinuousScheduler` `GRDenseContextKVPool` `GRHTTPServingAdapter` `GRServingWorker` |

另有 `src/gr_inference_trtllm_kernels/qwen3.py`（1513 行）作为**可选**的 JIT CUDA kernel 扩展，默认全关。

```
examples/sid-gr-inference/
├── src/gr_inference/              ← 框架本体（6 子包）
├── src/gr_inference_trtllm_kernels/ 可选 JIT CUDA kernel
├── examples/                      run_tiny_gr.py / run_tiny_serving.py（最小用法）
├── tools/                         真实权重 serving / benchmark / 比较工具
├── scripts/                       shell 包装：env → CLI
├── benchmarks/  tests/  docs/
├── Dockerfile  pyproject.toml
```

---

## 4. 核心抽象：KV 三件套 + GRGenerationState

### 4.1 KV 三件套（`gr_kv/`，框架的「原生抽象」）

| 抽象 | 内存布局 | 角色 |
|---|---|---|
| **ContextKV** (`context_kv.py:14`) | `[layers, B, S_ctx, Hkv, D]`，dense 连续 | 一个请求的**长 prefill context**，所有 beam 共享，**prefill 只写一次** |
| **BeamKV** (`beam_kv.py:24`) | `[layers, B, max_decode_steps, max_beam_width, Hkv, D]`，**step-major** | 短的每 beam decode 历史；可零拷贝 reshape 成 kernel 要的 `[B, steps·W, Hkv, D]` |
| **BeamPath** (`beam_path.py:12`) | 纯元数据：每步一组 `(parent_beams, token_ids, scores)` 平行 tuple | **逻辑** parent→child 血缘；`token_trace(beam)` 回溯重建完整序列 |

> 设计要点：**BeamKV 在 decode 之间从不物理重排**。beam 剪枝/扩展只改 BeamPath 元数据；被剪 beam 的 KV 留在 BeamKV 里但不再被引用。血缘在 attention kernel 内由 `topk_indices` gather 解析（§6）。这是「短 decode」才能成立的布局。

### 4.2 中央状态对象 `GRGenerationState`（`gr_runtime/generation.py:37`）

把 prefill 输出桥接到 decode，是整个离线路径的中枢：

```python
generation = GRGenerationState.from_prefill(
    request_id=..., prefill=prefill,
    max_decode_steps=..., max_beam_width=...,
    fixed_beam_width=..., beam_score_mode="raw_logits" | "logprob",
)  # 内部：按 ContextKV 形状 allocate BeamKV；新建 max_decode_steps+1 的 BeamPath
generation.initialize_beams(item_mask=...)        # prefill 末位 logits 选初始 topK → BeamPath step0
generation.update_beams_from_logits(logits, ...)  # 每 decode 步：select_next_topk → 追加 BeamPath
```

- `from_prefill` 用 `allocate_beam_kv_like_context` 让 BeamKV 继承 ContextKV 的 layers/heads/dim/dtype/device。
- 两种打分模式：`raw_logits`（累加原始 logits，**离线默认**）vs `logprob`（对数概率，**serving 默认**）——两条路径默认不一致（§12）。

---

## 5. 执行分层：离线库 / 同步 serving / 连续拼批 serving

同一套 `Qwen3GRModel` + KV 抽象之上，叠了三层由简到繁的执行器：

### 第 1 层：离线库（单请求，同步）— `examples/run_tiny_gr.py`
```
Qwen3GRModel.forward_prefill → GRGenerationState.from_prefill
→ model.generate_fixed_beam(generation, GRDecodeEngine(...), max_steps, item_mask_provider, beam_width_policy)
```
驱动者 `FixedBeamDecodeLoop.run`（`decode_loop.py:61`）：逐步「写 BeamKV → 跑 decode attention → logits → topK → 追加 BeamPath」，按策略收缩宽度，遇 stop token / item_complete 停。

### 第 2 层：同步 serving（多请求，仍同步）— `examples/run_tiny_serving.py`
```
GRServingEngine(model, decode_engine, GRServingConfig)   # 批量 prefill + 可选批量 decode
+ SyncGRScheduler(engine, SchedulerPolicy)              # FIFO 拼批骨架
scheduler.submit(request) × N → scheduler.run_until_empty()
```
`SyncGRScheduler`（`queue.py`）自称 "skeleton"，是第 3 层的简化前身；`engine.py` 的 `enable_batched_decode` 默认关。

### 第 3 层：连续拼批 serving（生产路径）— `tools/serve_qwen3_gr_http.py`
```
GRContinuousScheduler (调度状态机) + GRContinuousServingExecutor (模型执行器)
     ↑ driven by
GRServingWorker (后台 daemon 线程，循环 facade.tick())
     ↑ wrapped by
GRInProcessServingFacade (纯 Python 控制面) ← GRHTTPServingAdapter (JSON↔facade) ← http.server
```

三层共用同一个模型与 KV 抽象——**复杂度是叠加的，不是替换的**。

---

## 6. kernel 作为可插拔后端：框架↔kernel 契约

这点最值得讲清，因为它正是「本目录自成体系」的体现。

- **框架侧 wrapper 故意不依赖 torch**：`gr_kernels/attention/gr_decode_attention.py` 只 import `TensorSpec`/`shape_of`（来自 `gr_kv/layouts.py`），不 import torch。它定义纯契约 `GRDecodeAttentionInputs` + 校验器 + `KernelBackend = Callable[[Inputs], Any]`。
- **`BeamPath` 不进 kernel**：`GRDecodeAttentionInputs` 虽带 `beam_path`，但**只用于校验**（`steps_done`）。真正传给 kernel 的是 `topk_indices`。
- **唯一的 tensor 桥**是 `existing_kernel_backend.py` 的 `ExistingGRDecodeAttentionBackend.__call__`：
  - Q：`[B,W,Hq,D] → unsqueeze → [B,1,W,Hq,D]`
  - ContextKV：取层 → `[B,S_ctx,Hkv,D]`
  - BeamKV：切片 `[:, :decode_nums, :active_beam_width]` + reshape → `[B, decode_nums·W, Hkv, D]`
  - 调外部 `interface.beam_decode_attn(q, k_ctx, v_ctx, k_beam, v_beam, topk_indices, decode_nums, backend="dsl")`
- **kernel 动态加载**：`_load_kernel` 把 `kernel_root` 加到 `sys.path` 后 `importlib.import_module("interface")`。`kernel_root` 解析顺序：显式参数 → `GR_DECODE_ATTEN_ROOT` → `third_party/gr-decode-attention` → `$WORKSPACE/gr-decode-attention`。
  - ⚠️ **实证**：本仓 `.gitmodules` 里**没有** `gr-decode-attention`（只有 cutlass/FBGEMM/FlexKV），`third_party/gr-decode-attention` 目录为空。所以全新 checkout 下**默认连不上 real kernel**，需手动 clone 上游或把 `GR_DECODE_ATTEN_ROOT` 指向 `corelib/gr_decode_atten`。

### `topk_indices` 怎么造（`gr_runtime/batched_topk_indices.py:14`）

对每个 query beam 沿 `parent_beams` 链**逆向回溯到根**，写入 `step·W + ancestor`，最终形状 `[B,1,Hq,decode_nums,W]`。当 beam 宽度会收缩、祖先落到 active width 之外时，连续拼批路径改走 `make_compacted_batched_topk_indices`：先 `compact_batched_beam_kv_history` 把每 beam 的祖先物理 compact 到自己的槽，index 退化为平凡 `arange`。

> 对 kernel 内部算法（context MMA + 稀疏 beam gather + log-sum-exp 合并）的详解见 §10；本框架只负责「把正确形状的 tensor 喂给它」。

---

## 7. serving 内部架构

### 7.1 连续拼批调度（`continuous.py`，3562 行，系统心脏）

`GRContinuousScheduler.tick`（`continuous.py:379`）每 tick 顺序做四件事：

1. **超时清理** `_fail_timed_out_requests`：`tick_count - submitted_tick > timeout_ticks` → failed。
2. **准入 prefill** `_admit_prefill_batch`（:512）：从 `waiting_prefill` 弹至多 `max_prefill_batch_size` 个；每个经 `_can_allocate_kv`（KV lease 预算）门控——通过则分配 lease、设 `active_beam_width`、`stage="decoding"`；预算不足则 defer；若此时无任何在跑则 raise。
3. **跑 prefill**（executor 回调 `_run_prefill`）。
4. **规划并跑 decode**：`_plan_decode_batches`（:544）→ `_run_decode_batches`（:1690）。

**拼批单元 = 请求 × 活跃 beam**：decode 批按 `(decode_step, active_beam_width, next_beam_width, context_len)` 四元组分组（`continuous.py:547`），再按 `max_decode_batch_size` 切块。同组共享一次 tensor launch。status 里直接广告 `"decode_batch_grouping": "step,active_beam_width,next_beam_width,context_len"`。

**tensor-selection 快路径**（`_run_decode_batch_tensor_selection`:1864，默认开）：无 item 约束/策略/stop token 时，beam selection 全程留 GPU tensor，避免逐步 CPU 同步，结束时才 materialize BeamPath。被 `GR_INFERENCE_DISABLE_DECODE_TENSOR_SELECTION` 或 `return_beam_details` 关闭。

### 7.2 内存池（`memory.py`）

- `GRDenseContextKVPool` / `GRDenseBeamKVPool`：真 dense 大 buffer，每请求拿一段连续 slot（零拷贝 view）。带 lease、capacity、high-water、利用率、**泄漏检测**（`continuous.py:710` 用集合差检查 active 状态 vs 当前 lease）。
- `GRKVLeaseAllocator`：metadata-only 预算追踪（`max_running_requests`/`max_context_tokens`/`max_beam_slots`）。
- `GRPagedKVLeaseAllocator`：metadata-only、**无 tensor 后端**——page-backed KV 是明确 TODO。

### 7.3 CUDA graph（架构最有味道处）

- **decode graph 绑定稳定池 slice**：replay 复用请求自有 KV 地址、零拷贝。缓存 key 把 4 个 KV tensor 的 `data_ptr+storage_offset` 都算进去（`decode_cuda_graph.py:334`）+ pointer guard 防漂移，LRU + entry limit（默认 128）。
- **pool-view 机制**（`_contiguous_pool_view:3355`）：多请求 KV view 共享同一 storage 且 slot 连续 → `as_strided` 零拷贝拼批张量；否则退化 `torch.cat`；窗口不连续 → `dynamic_skip` 退 eager（计入 `decode_cuda_graph_dynamic_skips`）。
- **bucket 对齐**：实际 batch 向上取整到 warmup 过的 bucket（默认 1,2,4,8），补零后切片回真实 batch。
- **piecewise prefill graph**：切 `embed + 若干 layer chunk + output`（Qwen3-1.7B 28 层 / 4 chunk = 6 段）。
- **freeze-after-warmup**：warmup 后 `freeze_captures()`，之后 miss 只 replay、不新捕获。

### 7.4 prefix cache（`prefix_cache.py`）

radix 树实现，但**默认关**（`prefill_cache_enabled=False`）：GR 还没有 paged KV 分配器，命中只能整 PrefillResult clone（自述「still useful for exact hits」）。

### 7.5 beam 策略 / item 约束（推荐系统特有）

- **策略**（`gr_scheduler/beam_policy.py`）：`FixedBeamPolicy` / `ScheduledBeamPolicy`（step→width 时间表）/ `ScoreMarginBeamPolicy`（自适应：保留与最佳 score 差距 < margin 的 beam，默认只缩不涨）。
- **item 约束**（`gr_runtime/item_constraints.py`）：推荐场景下「token = semantic ID」，生成必须只产出合法 item 序列。`TokenTrie` 构造 `[W,vocab]` 合法掩码，topK 在掩码后 logits 上做；`SemanticItemCatalog` 支持热加载/回滚/版本化。

---

## 8. 服务启动详细过程

启动分两层：**shell 包装**（`scripts/serve_qwen3_gr_http.sh`）把环境变量翻译成 CLI，**Python 入口**（`tools/serve_qwen3_gr_http.py`）组装对象链并起 HTTP。

### 8.1 shell 包装

1. `source common_paths.sh`，定 `REPO_ROOT`。
2. 解析模型来源：`MODEL_DIR ← GR_MODEL_DIR ← 默认本地目录`；都没有则 `MODEL ← HF repo id`。
3. **定 kernel 根**：`export GR_DECODE_ATTEN_ROOT="${GR_DECODE_ATTEN_ROOT:-$(gr_default_decode_atten_root)}"`（`common_paths.sh:100`）。
4. 读约 40 个 `GR_*` env 变量赋默认，拼成 `args=(--flag value ...)`。
5. `gr_append_*_if_env_set` 把可选 env（catalog、prefill cache、suppress token、api key…）按需追加。
6. `exec python tools/serve_qwen3_gr_http.py "${args[@]}" "$@"`。

> `GR_DECODE_BACKEND=real`（默认）由 `serve_qwen3_gr_http.sh:32` 读，转成 `--decode-backend real`（:54）。**env 是入口，CLI 才是被 Python 消费的形式**（§12）。

### 8.2 Python 入口 `main()`（`serve_qwen3_gr_http.py:738`）

```
args   = build_parser().parse_args()
adapter = build_http_serving_adapter(args)   # 组装对象（8.3）
server  = adapter.serve(host, port)          # 起 ThreadingHTTPServer
print(routes...); server.serve_forever()
```

### 8.3 `build_http_serving_adapter`（:47）对象组装链

| # | 构造物 | 作用 |
|---|---|---|
| a | `_normalize_args` | 强制 `continuous=True`、`batched_decode=True`，补 warmup/freeze 默认 |
| b | `load_model` | 解析 HF/本地 ckpt → `materialize_qwen3_checkpoint` → `Qwen3GRModel` |
| c | `GRDecodeEngine(GRDecodeAttention(backend=make_decode_backend(...)))` | `real`→`ExistingGRDecodeAttentionBackend()`（按 `GR_DECODE_ATTEN_ROOT` 动态 import 外部 `interface.beam_decode_attn`）；`fake`→恒等 lambda |
| d | `GRServingEngine(model, decode_engine, GRServingConfig(..., enable_batched_decode=True))` | 同步模型执行器 |
| e | `_make_beam_kv_pool` / `_make_context_kv_pool` | dense 池（BeamKV/ContextKV 大 buffer） |
| f | **（若 warmup）`_warmup_online_shapes` + `_freeze_cuda_graph_captures`** | 见 8.4 |
| g | `GRContinuousScheduler(GRContinuousBatchingPolicy(...))` | 真正的连续拼批调度状态机 |
| h | `GRContinuousServingExecutor(engine, scheduler, pools, ...)` | 把 scheduler 回调接到模型执行/池/graph runner |
| i | `GRInProcessServingFacade(executor, item_mask_provider_store=catalog)` | transport-free 控制面 |
| j | `serving = GRServingWorker(facade, ..., autostart=True)` | **起后台 daemon 线程 `gr-serving-worker`** |
| k | `GRHTTPServingAdapter(serving, request_factory, validation_policy, ...)` | JSON↔facade HTTP 适配器 |

### 8.4 启动 warmup（:199，关键且易漏）

`_warmup_online_shapes` **单独**新建一个临时 `executor+scheduler`（:210）来驱动**同一个 engine**，目的是把 CUDA graph 捕获进 engine 持有的 graph runner：

- 枚举 warmup case：`(first_slot, batch_size)` 池窗口组合，最多 `warmup_online_max_cases`（默认 64）。
- 每个 case：必要时先灌 `first_slot` 个 blocker 请求只跑 prefill（占住对应池 slot lease，:274），再灌 `batch_size` 个 target 请求（`ignore_eos=True` + special-token suppress，与真实 `/generate` 同路径），`run_until_empty` → 触发 prefill/decode CUDA graph 捕获到**生产 batch bucket 与池窗口**。
- warmup 后 `_freeze_cuda_graph_captures(engine)`（:327）：对 engine 的两个 graph runner 调 `freeze_captures()`，之后只 replay。

> 生产 scheduler/executor（g/h）在 warmup 之后才建，但与 warmup **共用同一个 engine**，所以 warmup 捕获的 graph runner 在生产路径直接复用。这是 README「30 次 decode 捕获不再增长」的来源。

### 8.5 起服务（`adapter.serve`，http.py:143）

```
make_http_handler(self)                          # 闭包出 BaseHTTPRequestHandler 子类
ThreadingHTTPServer((host,port), handler_cls)    # 每个请求一个线程
server.serve_forever()
```

此时系统有**两类线程**（理解请求处理的前提）：
- **HTTP 线程**（每请求一个）：解析请求 → `adapter.handle` → `_dispatch`。
- **worker daemon 线程** `gr-serving-worker`（`worker.py:225 _run`）：循环 `_drain_pending_submissions → facade.tick()`，是真正驱动 prefill/decode 的引擎。

---

## 9. 请求处理全流程

以生产模式（后台 worker 开、manual tick 关）下 `POST /generate` 为例。

### 阶段 A：HTTP 接收 + 入队（HTTP 线程）

1. `ThreadingHTTPServer` 收连接 → handler → `adapter.handle("POST", ("generate",), body)`。
2. `handle`：`_validate_auth` → `_validate_body_size` → 解析 JSON → `_dispatch`（http.py:153）。
3. 路由到 `_handle_sglang_generate(payload)`（http.py:225→368）：
   - `_validate_admission(1)`：`waiting_prefill` 超 `max_waiting_requests` → **429 overloaded**。
   - `_sglang_generate_payload_to_gr_payload`：SGLang → 内部 schema。**必须 `input_ids`（拒 `text`）**；`beam_width=sampling_params.n`；`max_decode_steps=max(1,max_new_tokens-1)`；`ignore_eos/stream` 进 metadata。
   - `_make_request` = `make_torch_request_factory`（:158）：`input_ids` 转 long tensor（**放 CPU**）；`ignore_eos` 且配置了 suppress → 自动追加 `token_suppress` logits processor；构造 `beam_width_policy` 等。
   - `_validate_request`：decode_steps/beam_width/context_len/timeout_ticks 上限 → **400**。
   - **`facade.submit(request)`**（api.py:27）：`_prepare_request` 挂 item_mask_provider（若 catalog）→ 校验 + 唯一性 → `executor.submit` → `scheduler.submit`（continuous.py:300）：压入 `waiting_prefill`，建 `GRContinuousRequestState`。
   - **`_wait_for_response(request_id)`**（http.py:377）：**busy-poll** `facade.poll(id)` 查 `scheduler.finished`，1ms 间隔；若 `allow_manual_tick` 且 worker 未运行 → `facade.tick()` 内联；**300s → 504**。
   - HTTP 线程**就此阻塞轮询**，等 worker 把结果写进 `finished`。

### 阶段 B：worker 驱动调度（worker 线程）

4. `GRServingWorker._run`（worker.py:225）循环：`_should_wait_for_batch_fill`（攒批）→ `_drain_pending_submissions`（一次性 submit_many）→ `_tick_unlocked` → `facade.tick()` → `executor.tick()`；忙时 `tick_interval_s=1ms`、闲时 `idle_sleep_s=5ms`、异常 `error_sleep_s=50ms`。

5. `executor.tick()`（continuous.py:984）= `scheduler.tick(prefill_executor=_run_prefill, decode_executor=_run_decode_batches)`，一个 tick 做四件事：

   **(B1) 超时清理**。

   **(B2) 准入 prefill** `_admit_prefill_batch`：弹至多 `max_prefill_batch_size` 个，KV lease 预算门控，通过则 `stage="decoding"` 移入 `self.decoding`。

   **(B3) 跑 prefill** `_run_prefill`（:1167）：按 `input_ids.shape` 分组；每组 `_run_uncached_prefill_requests`：`torch.cat(input_ids).to(device)`，从池取连续 slot 窗口，`_forward_prefill`（优先 prefill CUDA graph，否则 `model.forward_prefill(last_token_logits_only=True)` 写 ContextKV + 返末位 logits）；`_store_prefill_results`：批量初始 topK → 切片成 per-request `PrefillResult` → 建每请求 `GRGenerationState`（分配 BeamKV、建 BeamPath、种 step 0）。

   **(B4) 规划并跑 decode**：
   - `_plan_decode_batches`：按四元组分组切块 → `GRContinuousDecodeBatch` 列表。
   - `_run_decode_batches` → 每 batch `_run_decode_batch`（:1706）：
     - 构造每请求池窗口视图（连续→`as_strided` 零拷贝，否则 `torch.cat`）。
     - 判 `_decode_cuda_graph_has_stable_pool_window`：稳定且命中 bucket → graph replay；否则 `dynamic_skip` 退 eager。
     - **逐 decode step**（`forward_decode_step`:1968）至请求完成：
       1. `make_batched_beam_token_ids(selection)` → `[B,W]`。
       2. `make(_compacted)_batched_topk_indices` → `[B,1,Hq,decode_nums,W]`（按 BeamPath 血缘回溯祖先）。
       3. `model.forward_decode_step(...)`：逐层 QKV+QKnorm+RoPE → `BeamKVWriter.write_layer_step(step)` 写当前步 K/V → `GRDecodeAttention` → 经 `ExistingGRDecodeAttentionBackend` 切成 kernel 形状 → 外部 `beam_decode_attn(ContextKV + BeamKV[..., :step+1, :W], topk_indices)`。
       4. 末位 logits → item mask（trie 约束）→ `select_next_topk_batched` → 必要时 scatter BeamKV 回各槽 → 追加 BeamPath → `step++`。
   - 完成的请求 → `_finish_state`（:2254）默认附 **`beam_results`**（`beam_details` 仅 `return_beam_details` 开时）→ `_store_finished_response` 写入 **`scheduler.finished[request_id]`**。

### 阶段 C：HTTP 线程取走结果（HTTP 线程）

6. worker 写入 `scheduler.finished` 后，HTTP 线程下一次 `facade.poll(request_id)` 命中，`_wait_for_response` 返回。
7. `_sglang_generate_response`（http.py:592）组装 SGLang 形状 `{text:"", output_ids, meta_info:{beam_results, completion_tokens=beam_width·max_new_tokens, prompt_tokens, finish_reason, ...}}`。
8. **200 OK** 返回客户端，HTTP 线程结束。

### 时序图

```
HTTP 线程                         worker 线程 (gr-serving-worker)
────────────                      ───────────────────────────────
POST /generate
 → validate/admit
 → facade.submit ───────────────► waiting_prefill 队列
 → _wait_for_response              _run 循环:
     poll() ✗ (sleep 1ms)            drain pending → facade.tick()
     poll() ✗                        ┌─ tick ──────────────────────┐
     ...                             │ fail timeouts               │
                                     │ admit prefill batch (lease) │
                                     │ _run_prefill → forward_prefill
                                     │   (写 ContextKV, 末位 logits)│
                                     │ plan decode batches          │
                                     │ _run_decode_batches:         │
                                     │   per step:                  │
                                     │     write BeamKV[step]       │
                                     │     beam_decode_attn(...)    │
                                     │     topK → append BeamPath   │
                                     │ _finish_state → finished[id] │
                                     └─────────────────────────────┘
     poll() ✓ ←──────────────────   scheduler.finished[id]
 → sglang_response
 ← 200 OK
```

### 设计要点

- **HTTP 线程不碰 GPU**：生产模式下只 submit + 轮询；所有 prefill/decode 都在 worker 线程的 tick 里。`/generate` 的「同步」靠 busy-poll + 后台 worker 实现，而非 HTTP 线程自己跑模型。
- **同步兜底**：关后台 worker（`--disable-background-worker`）并开 `--allow-manual-tick` 时，`_wait_for_response` 会在轮询间隙自己 `facade.tick()`，退化成单线程同步服务。
- **300s 硬超时 → 504**。
- **prefill 按 shape 分组、decode 按 (step,beam,next_beam,ctx) 分组**——拼批粒度，决定同一 tick 能合并多少请求成一次 tensor launch。
- **CUDA graph 命中条件**：池窗口连续 + batch 命中 warmup 过的 bucket；否则 `dynamic_skip` 退 eager。

---

## 10. 外部 kernel：gr_decode_atten 算法简述

`corelib/gr_decode_atten/` 是内部仓库 `cjerry/gr-decode_atten` 的 vendored 快照（pinned `1c540f6`），CuTe DSL (CUTLASS) 实现，覆盖 SM8x/SM90/SM100/SM120。

### 公开 API（`corelib/gr_decode_atten/interface.py:835`）

```python
beam_decode_attn(
    q,              # [B, seqlen_q=1, W, Hq, D]            bf16/fp16
    k_context, v_context,   # dense [B, S_ctx, Hkv, D] 或 jagged [total_k, Hkv, D]
    k_beam, v_beam,         # [B, decode_nums·W, Hkv, D]
    topk_indices,   # [B, 1, Hq, max_decode_nums, W]  int32
    decode_nums,
    softmax_scale=None, backend="dsl",   # "dsl"=融合, "3kernel"=K1+K2+K3
    seqused_k=None, cu_seqlens_k=None,   # 仅 3kernel 支持
) -> (out [B,1,W,Hq,D], lse [B,1,W,Hq])
```
仅前向（`backward` 直接 raise）。

### 算法本质：把 attention 拆成「共享 dense 区 + 每 beam 稀疏区」

beam decode 时 KV 天然分两块（`tests/reference.py:307` 的 golden reference 证实）：

- **Context KV**：prefill 产出，**所有 beam 共享**，dense 顺序 → tensor core MMA。
- **Beam KV**：decode 产出，**每 beam 独立**，topK 不规则 gather → CUDA core FMA。

两条路径：

**3-kernel 流水线**（SM100/Blackwell 强制走这条，融合 beam 在 B200 上更慢）：
```
K1: Context Attention   (tensor core MMA + split-KV)  → ns 个 fp32 partial
K2: Beam Sparse Attention(CUDA core FMA, topK gather) → 1 个 fp32 partial
K3: Combine             (log-sum-exp 合并 ns+1 partial) → bf16 输出
```

**融合路径**（SM8x/SM90/SM120 默认）：把 K2 **融合进 K1 的最后一个 split**，直接复用同一个 MMA 累加器 `acc_O` 和同一份 online-softmax 状态。

### topk_indices 的工作方式（设计的灵魂）

BeamKV 在 kernel 里 flatten 成 `[B, decode_nums·W, Hkv, D]`，flat 索引 = `step·beam_width + beam_slot`。`topk_indices[b,0,h,s,q]` = 当前 query beam `q` 在历史第 `s` 步应读的**祖先 beam flat 索引**。

- K2（`src/decode/flash_fwd.py:196` `_gather_load_tile`）：每 CTA 负责一个 (batch, beam, kv_head)，按 `kv_idx = gTopk[...]` 做**散落式全局读取** `gKV[kv_idx, col]`，无顺序扫描。
- **融合 beam 阶段**（`src/sm90/flash_fwd.py:1641`、`src/sm80/flash_fwd.py:1252` `_beam_sparse_phase`）：只在最后一个 split 跑；遍历 tile_m 累加器每行（每行 = 一个 beam），该 beam gather 自己的 `decode_nums` 行 KV，用 4 线程并行标量点积（HGMMA 累加器 TV 布局 4 线程共享一行）算 QK，**直接在同一个 `acc_O`/`row_max`/`row_sum` 上做 online softmax 更新**。Q 复用 context 阶段已加载的 SMEM Q tile。

### 为什么能赢 naive 每 beam 独立 decode

1. **Context 只算一次**：所有 beam 作为 query（M 维 = W）共同对同一条 context KV 做一次 MMA。
2. **Split-KV 撑满 SM**：decode 时 M 维小，沿 N(KV) 轴切分（`num_splits_heuristic`，移植自 FA-Hopper），再由 K3 合并。
3. **按区域选对计算原语**：dense context（千 token）用 tensor core；稀疏 beam gather（≤16 行）用 CUDA core，省掉把不规则 K 重排进 MMA A-operand 的开销。
4. 融合路径 `ns==1` 时**只需 1 次 kernel launch**直接出 bf16。

效果（README 性能分解表，机制一致）：同一 case decode attention kernel 时间 **SGLang 35.2ms → SID-GR 1.6ms**。

### 已知坑（代码实证）

- split-KV + `seqused_k` 在 SM90 H100 PCIe 会 hang，强制 `ns=1`（`interface.py:698` FIXME）；jagged(`cu_seqlens_k`) 也强制 `ns=1`。
- K2 用 `-5e4` 而非 `-inf` 作 sentinel，避免全 padding 时 `exp2(-inf- -inf)=NaN`（`src/decode/flash_fwd.py:325`）。
- 融合 beam 阶段只有 4 个共享一行的线程中 **1 个**写 `row_sum`（`is_row_sum_writer = t0==0`）。

---

## 11. 运转方式：环境变量 + shell + 容器

**几乎全靠环境变量驱动**：代码里出现 **121 个** `GR_*` / `GR_INFERENCE_*` 环境变量，覆盖：模型/设备（`GR_MODEL_DIR` `GR_DEVICE`）、负载（`GR_CONTEXT_LEN` `GR_BEAM_WIDTH` `GR_DECODE_STEPS` `GR_MAX_BATCH_SIZE`）、kernel（`GR_DECODE_BACKEND` `GR_DECODE_ATTEN_ROOT`）、池容量（`GR_BEAM_KV_POOL_CAPACITY` `GR_CONTEXT_KV_POOL_CAPACITY`）、CUDA graph（`GR_INFERENCE_DECODE_CUDA_GRAPH_*` `GR_FREEZE_CUDA_GRAPHS_AFTER_WARMUP`）、TRT-LLM JIT（`GR_INFERENCE_GR_TRTLLM_*`）、HTTP（`GR_HTTP_HOST/PORT`）、warmup（`GR_WARMUP_ONLINE_*`）等。

`scripts/`（13 个 shell）把这些 env 翻译成 CLI + `PYTHONPATH`。

容器化（`Dockerfile`）：基镜像 `pytorch/pytorch:2.6.0-cuda12.4`，`pip install -e ".[kernels]"`，入口 `scripts/serve_qwen3_gr_http.sh`，暴露 8000。

### TRT-LLM JIT kernel（可选，默认关）

`src/gr_inference_trtllm_kernels/qwen3.py`（1513 行）是手写 CUDA，`torch.utils.cpp_extension.load_inline` **JIT 编译**（非 Triton），注册 `torch.ops.gr_trtllm`：`fused_qk_norm_rope`（原地改写 packed QKV）、`gated_mlp`、`packed_gemm`（cuBLAS）+ 6 个 kernel。每个单独 env 开关，**默认全关**，属实验性优化路径，不在默认 hot path。

主线 hot path：gr_decode_atten(real) + FlashAttention(prefill) + torch eager 投影/norm/rope + 可选 flashinfer。

---

## 12. 文档说法 vs 代码实证

| 说法 | 代码核对 | 结论 |
|---|---|---|
| kernel「直接接收 ContextKV+BeamKV+BeamPath」 | `beam_path` 只用于校验；kernel 收 `topk_indices` | ❌ 不精确 |
| `GR_DECODE_BACKEND=real` | 仅 `serve_qwen3_gr_http.sh:32` 读它转 `--decode-backend`，Python 不读 | ✅ 成立（经 shell） |
| dense 池 + lease + 高水位 + 泄漏检查 | `memory.py` + `continuous.py:710/2506` | ✅ 属实 |
| pool-view decode graph、动态非连续退 eager | `_contiguous_pool_view` + pointer guard | ✅ 属实 |
| `beam_results` 默认 / `beam_details` debug | `continuous.py:2304` 恒返 / 受门控 | ✅ 属实 |
| 性能：ctx5000/batch8 离线 2.23×、在线吞吐 1.85×、decode attention 35ms→1.6ms | 与 kernel 设计（context 算一次 + 稀疏 gather）一致 | ✅ 机制可信，数字需复现 |

**代码发现的、文档没说的：**

1. 全新 checkout 默认**连不上 real kernel**（`third_party/gr-decode-attention` 不是 submodule 且为空）。
2. 离线默认 `raw_logits` 打分、serving 默认 `logprob`——同一框架两条路径**默认不一致**。
3. kernel wrapper **故意 torch-agnostic**（只依赖 `layouts.py` 的 `TensorSpec`），真正碰 tensor 的只有 `existing_kernel_backend.py` 一处——契约/实现分离很干净。
4. beam selection（log_softmax+topK）**仍在 decode CUDA graph 之外**（README §10 TODO 承认）。
5. Dockerfile 基镜像是 `pytorch:2.6.0-cuda12.4`，而 README 快速上手写默认镜像是 `lmsysorg/sglang:dev-cu13`——两套部署口径不一致。
6. `engine.py` 批量 decode 默认关、`queue.py` 的 `SyncGRScheduler` 是被取代的「骨架」——`continuous.py` 才是生产路径。
7. `continuous.py:1574` 有一行孤立 `pass`（脚手架残留，目前无害）。

---

## 13. 现状、局限与 TODO

- **单节点 alpha**：核心路径完整，剩生产化/广验证/可维护性。
- **多 GPU/scale-out**：当前单节点单 GPU；TP/PP、多副本、跨 GPU KV 与 beam 所有权是未来工作。
- **ContextKV 内存策略**：当前 dense；计划多 context bucket + page-backed 存储 + decode attention 内原生 page table。
- **beam selection 进 graph**：decode forward 已在 graph，log_softmax+topK+selection 仍在外。
- **prefix cache**：默认关，受限于无 paged KV 分配器。
- **item 约束**：trie/mask/constrained topK/热加载已有，待真实大目录验证。

---

## 14. 一句话总结

这是一个**为「长 context + 短 decode + 超大 beam」特化、且 kernel 可插拔**的自包含 GR 推理 + serving 框架。它的内部骨架是：

> **KV 三件套（ContextKV/BeamKV/BeamPath）** 作为原生抽象 → **GRGenerationState** 桥接 prefill↔decode → **三层叠加执行器**（离线库 / 同步 serving / 连续拼批 serving）→ 顶端 **HTTP /generate**。唯一的外部依赖是 **decode attention kernel**，通过注入式 `GRDecodeAttention(backend=...)` 解耦；框架自己负责造 `topk_indices` 把 beam 血缘喂给 kernel，从而**彻底避免 KV 在 beam 剪枝时的物理重排与 context 的重复读取**。

`corelib/gr_decode_atten` 只是那个被默认选用的 real backend 的实现，不在本目录代码边界内。
