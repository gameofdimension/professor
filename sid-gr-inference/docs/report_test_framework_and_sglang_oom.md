# SID-GR 推理测试框架与 SGLang OOM 调查报告

> 范围:剖析 SID-GR vs SGLang 离线基准的测试脚本链路,并记录一次 SGLang 大 beam
> 下 OOM 的根因调查与修复。模型 Qwen3-1.7B,硬件 NVIDIA RTX PRO 5000 72GB ×8。
> 产物目录:`benchmark_artifacts/sglang_compare/offline_decode_steps_*/`。

---

## 1. 背景

SID-GR(Semantic-ID 生成式推荐)推理的负载特征是 **长 context + 短 decode + 大 beam**:
用户历史很长(prefill 贵),cluster 深度浅(decode 只有几步),为多样性需要 beam=128/256/512/1024。
这与 vLLM / SGLang / TRT-LLM 优化的"多请求、长 decode、chat API"场景不同。本项目自研
了专门针对该负载的紧凑引擎,并用一套离线基准和**最强开源基线 —— SGLang 的 beam-search
PR 分支(`feature/beam_search`)**正面对比,验证自研引擎的价值。

本次工作:① 梳理测试脚本链路;② 新增"decode-steps × beam-width"二维扫描;③ 调查并修复
SGLang 在大 beam 下的 OOM。

---

## 2. 测试框架剖析

### 2.1 调用链总览

```
run_container.sh                     起 docker 容器(GPU/挂载/env 透传)
  └─ quickstart_offline.sh           编排:bootstrap → perf → accuracy
       ├─ run_offline_perf_benchmark.sh     [性能]   标量 GR/SGLANG_DECODE_STEPS
       │    └─ run_gr_sglang_perf_sweep.sh  ctx × beam × batch 三重循环
       │         ├─ make_qwen3_beam_workload.py    生成确定性 workload(JSONL)
       │         ├─ run_qwen3_real_weight_serving.py   [GR 引擎]
       │         ├─ run_sglang_beam_benchmark.py       [SGLang 引擎]
       │         └─ summarize_gr_sglang_perf_sweep.py  汇总 summary.md/csv
       └─ run_offline_accuracy_benchmark.sh [精度]
            ├─ (同三工具,加 --return-beam-details --record-outputs)
            ├─ compare_gr_sglang_beam.py     逐请求 token 比对
            └─ 内联 python 生成 accuracy summary
```

本次新增的外层驱动(见 §5):
```
run_decode_steps_perf_sweep.sh      decode × beam 双重外循环(每对独立 OUT_DIR,容错)
  └─ run_offline_perf_benchmark.sh  (复用,每次传一对 decode-steps + 单个 beam)
```

### 2.2 容器层 — `scripts/run_container.sh`

- 镜像 `lmsysorg/sglang:dev-cu13`(`run_container.sh:11`),`--gpus all` + `--shm-size 32g` + `--ipc=host`(`:97-124`)。
- 挂载四卷:repo、`models/`、`.cache/`、SGLang checkout(`:120-123,162-164`)。
- 白名单 env 透传(`:126-156`):`MODEL / CONTEXT_LENS / BEAM_WIDTHS / BATCH_SIZES / REPEAT / GR_DECODE_STEPS / SGLANG_DECODE_STEPS / RUN_ACCURACY / ...`。`DOCKER_RUNTIME=""` → 不加 `--runtime`,只用 `--gpus`(`:158-160`)。
- 注:脚本用 `-it`,非交互后台执行需用 `script -qec` 包一层伪终端,否则 docker 报 `the input device is not a TTY`。
- `DOCKER_USER` 默认当前 uid:gid;`PYTHONUSERBASE=/workspace/.cache/python_user`,所以 bootstrap 安装的 kernel 落在挂载缓存里,**容器重启后复用**。

### 2.3 编排层

- `scripts/quickstart_offline.sh`:可选地 `RUN_ACCURACY=1` 串起 perf + accuracy;`SKIP_BOOTSTRAP=1` 可跳过 kernel 安装。
- `scripts/run_offline_perf_benchmark.sh`(**性能入口**):设置公平性开关并调 `run_gr_sglang_perf_sweep.sh`(`:84-87`):
  ```
  SGLANG_DISABLE_RADIX_CACHE=1   关 SGLang radix/prefix cache(避免命中作弊)
  GR_ENABLE_PREFILL_CACHE=0       关 GR prefill cache(同上)
  GR_RETURN_BEAM_DETAILS=0        性能阶段不收集 beam 明细(省开销)
  ```
- `scripts/run_offline_accuracy_benchmark.sh`(**精度入口**):同样的 cell 循环,但 GR/SGLang 都带 `--return-beam-details --record-outputs`,然后 `compare_gr_sglang_beam.py` 逐请求比对。

### 2.4 扫描层 — `scripts/run_gr_sglang_perf_sweep.sh`

```bash
for context_len in ${CONTEXT_LENS}; do
  for beam_width in ${BEAM_WIDTHS}; do
    for requests in ${BATCH_SIZES}; do
      生成 workload → 跑 GR(--decode-steps ${GR_DECODE_STEPS}) → 跑 SGLang(--decode-steps ${SGLANG_DECODE_STEPS})
    done
  done
done
summarize_gr_sglang_perf_sweep.py ${OUT_DIR}
```
- **每个 cell 重新起一次引擎**(GR/SGLang 各自 warmup + REPEAT 次测量),避免状态泄漏,代价是每个 cell 有 ~30-60s 启动 + CUDA graph 抓取开销。
- 默认 `REPEAT=3`,性能主指标 `wall_ms_median`。
- ⚠️ `set -euo pipefail` + 每 cell `if ! ...; then exit 1; fi`:**任一 cell 失败会 abort 整个子扫描**,summary 在最后才写,所以失败 cell 之后的所有 cell(哪怕本来能跑)全丢。这是 §3 调查中 b512/b1024 数据残缺的直接原因,也是本次新增外层 `run_decode_steps_perf_sweep.sh` 做"每对独立 OUT_DIR + 失败隔离"的动机。

### 2.5 工具层 — `tools/*.py`

| 工具 | 作用 | 关键产出 |
|---|---|---|
| `make_qwen3_beam_workload.py` | `--no-tokenizer` 下用**确定性 token id**(基于 request idx 的可重复伪随机,`:83-90`)生成定长 context。是合成负载,可复现可对比。 | `qwen3_ctx{N}_req{R}.jsonl` |
| `run_qwen3_real_weight_serving.py` | GR 引擎驱动:`GRServingEngine` + `GRContinuousScheduler` + `GRDenseBeamKVPool` / `GRDenseContextKVPool` + `gr-decode_atten` 真 kernel(`:34-62`)。池容量 `--beam-kv-pool-capacity ${requests}`。 | `gr_*.json`(含 `wall_ms_median / prefill_ms_median / decode_ms_median`) |
| `run_sglang_beam_benchmark.py` | SGLang 引擎驱动:`sgl.Engine(enable_beam_search=True)`,`--use-input-ids` 直喂 token,`arrival-mode batch`。 | `sglang_*.json`(含 `wall_ms_median / qps_median`) |
| `compare_gr_sglang_beam.py` | 精度比对:top1 exact / topK 集合重叠 / 有序前缀 / token 长度(`:63-73, 290-300`)。 | `compare_*.md/json` |
| `summarize_gr_sglang_perf_sweep.py` | 配对 `gr_*` / `sglang_*`,算 `speedup = sglang_wall / gr_wall`(>1 即 GR 胜,`:24-47`)。 | `summary.md/csv` |

### 2.6 关键设计:公平性
- 两引擎跑**同一份 JSONL**;关双方前缀缓存;`temperature=0` 贪心;`--no-tokenizer` 避免 tokenization 差异。
- 唯一口径差是 decode-steps 计数(见 §2.7),compare 工具显式建模、不强行对齐。

### 2.7 decode-steps 语义对齐 —— 为什么 GR=2 / SGLang=3 是对的

两引擎"步数"定义不同,但产出等长:

- **SGLang**:`--decode-steps` 直接 = `max_new_tokens`(`run_sglang_beam_benchmark.py:300`)。N 步 = N 个 token。
- **GR**:`max_decode_steps` 统计的是**初始 prefill beam 选择(step 0)之后**的 decode 迭代数。代码里有一个显式的 **step 0**(`continuous.py:1292 _select_initial_prefill_beams_batched`,`:1346/1359/1381 step=0`)产出第 1 个 token,之后 `for step in range(max_decode_steps)`(`engine.py:254`)再跑 N 步。所以总 token = `max_decode_steps + 1`(`run_qwen3_real_weight_serving.py:716-720`,`compare_gr_sglang_beam.py:179-182`)。

compare 工具的源码注释明确写了这点(`compare_gr_sglang_beam.py:77-78`):
> "GR max_decode_steps counts decode iterations after the initial prefill beam selection;
>  fixed-length GR outputs normally contain GR decode_steps + 1 token ids."

并用 `output_token_budget_match = (gr_budget(decode_steps+1) == sglang_decode_steps)` 校验(`:42-46`)。

**实测验证**(从 accuracy run 的 JSON 逐 beam 数):GR `max_decode_steps=2` → 每 beam 3 个 `output_ids`;SGLang `decode_steps=3` → 每 beam 3 个 `token_ids`;且 `token_length_match_rate=1.0`。所以 `2+1 = 3 = 3`,标定到同一实际输出长度。把两边都设 3 反而让 GR 产 4 个 token(多算一步、长度不齐)——**所以 2 vs 3 才是对的**。

---

## 3. SGLang 大 beam OOM 调查

### 3.1 现象
扫描 beam ∈ {256,512,1024} 时,SGLang 在 `batch × beam ≈ 4096` 处集体 OOM,边界极干净:

| beam | SGLang 跑通的 cell | OOM 点 |
|---|---|---|
| 256 | 全部 ctx × batch | 不 OOM |
| 512 | 仅 ctx1000 × {1,2,4} | ctx1000/batch8(512×8=4096)💥 |
| 1024 | 仅 ctx1000 × {1,2} | ctx1000/batch4(1024×4=4096)💥 |

1.7B 小模型在 72GB 卡上 OOM,明显反常 —— 用户提出需调查。

### 3.2 定位
SGLang 抛错栈:
```
schedule_batch_beam_search_mixin.py:196 _prepare_for_new_beam_search
  raise RuntimeError("Out of memory. Please set a smaller number for
                      `--max-running-requests` or `--beam-width`.")
```
对应代码(`schedule_batch_beam_search_mixin.py:191-198`):
```python
total_slots = sum(req.beam_width for req in new_reqs)   # = batch × beam_width
beam_req_pool_indices = self.req_to_token_pool.alloc_by_count(total_slots)
if beam_req_pool_indices is None:
    raise RuntimeError("Out of memory. ...")
```
→ 这是**槽位计数(index 池)**问题:`req_to_token_pool` 空闲槽不足,不是 KV 字节不够。

### 3.3 根因 —— 一个硬编码的 4096 上限
池大小来源(`model_runner_kv_cache_mixin.py:272-274`):
```python
self.req_to_token_pool = ReqToTokenPool(size=max_num_reqs, max_context_len=...)
```
而 `max_num_reqs` 的推导(`model_runner_kv_cache_mixin.py:698-717`,函数 `_resolve_max_num_reqs`):
```python
estimated = int(token_capacity / self.model_config.context_len * 512)
estimated = max(min(estimated, 4096), 2048)          # ← 硬上限 4096
max_num_reqs = self.server_args.max_running_requests
if max_num_reqs is not None:
    max_num_reqs = min(max_num_reqs // self.dp_size, estimated)   # ← 即便用户设了也被 min 下去
else:
    max_num_reqs = min(estimated, token_capacity // 2)
```
**`max_num_reqs` 永远 ≤ 4096**,与模型大小、显存、`--max-running-requests` 都无关 —— `min(estimated, 4096)` 这一行钳死了。beam search 一次要 `batch × beam` 个槽,所以 `batch × beam > ~4088`(池=4096,prefill 已占几个槽)必 OOM。这与 §3.1 的边界完全吻合。

### 3.4 为什么小模型也 OOM
- **不是权值显存**:1.7B 权值 ~3.4GB,72GB 卡绰绰有余(CUDA graph 抓取时实测还剩 ~12GB free)。
- **不是 KV 字节**:小模型 per-token KV 很小,KV 字节池很大。
- **是一个与模型无关的代码常量 `4096`**:它假设"每槽 = 一个用户请求",4096 并发对正常 chat serving 绰绰有余;beam search 把每个请求展开成 `beam_width` 个槽,直接打破假设。**所以无论模型多小、显存多大,`batch × beam > 4096` 就是过不去。**

### 3.5 验证过程(`tools/verify_sglang_beam_oom_fix.py`)
在容器里逐个验证:

| 配置 | 结果 | 结论 |
|---|---|---|
| DEFAULT 引擎,b512/batch8 | OOM(同 line 196) | 复现成功 |
| `max_running_requests=8192` 单独 | **仍 OOM** | 证明用户参数救不了,钳制是代码级 |
| 放宽 clamp + `context_length=6000` | 过了槽位 OOM,但撞第二堵墙(logits) | 见 §3.6 |
| 放宽 clamp + `context_length=6000` + `mem_fraction_static=0.65` | **b1024/batch8(8192 beam)SUCCESS** | 三合一修复通过 |

### 3.6 第二堵墙 —— 全词表 logits
放宽 clamp 后,b1024/batch8 撞到 `_compute_lm_head` 的 CUDA OOM:
```
logits_processor.py:913 _compute_lm_head   logits = torch.matmul(...)
torch.OutOfMemoryError: Tried to allocate 2.32 GiB. GPU 0 ... 70.11 GiB in use.
```
原因:SGLang 对每条 beam-sequence 算**全词表(151936)logits**,张量 `[batch × beam, vocab]`。b1024/batch8 = 8192 × 151936 × 4B ≈ **5GB**,加上默认 KV 池吃了 ~60GB,没给它留空间。降 `mem_fraction_static` 腾出显存即可。

> 这是 GR 的**第二个结构性优势**:GR 是 item-constrained decode,只算候选小词表,不存在这个全词表 logits 膨胀。

---

## 4. 修复方案

三处改动,缺一不可(数学上互补):

| 改动 | 位置 | 作用 |
|---|---|---|
| 放宽 4096 钳制 | `model_runner_kv_cache_mixin.py:703` | `max(min(estimated,4096),2048)` → `max(estimated,2048)`。去掉上限。**安全**:index 池大小 = `estimated × context_len × 8B = token_capacity × 512 × 8` ≈ 2.3GB(与 context_len 无关地自限),不会失控 |
| `context_length=6000` | `run_sglang_beam_benchmark.py` 新增 `--engine-context-len` → `sgl.Engine(context_length=...)` | 模型默认 40960 把 `estimated = token_capacity/context_len×512` 压得很小;改 6000 后 estimated 升到 ~47513(仅放宽 clamp 不改 context_len 只到 ~6957,<8192,不够 b1024/batch8) |
| `mem_fraction_static=0.65` | 同上 `--mem-fraction-static`(已存在) | 给全词表 logits matmul 腾显存(§3.6) |

`context_len=6000` 的取值:模型默认 40960,但本负载最大 ctx=5000、decode 预算 +5,所以 6000 足够(>5005)。

### 接入方式 —— 环境变量驱动,非破坏性
- `scripts/run_gr_sglang_perf_sweep.sh`:用 `common_paths.sh` 的 `gr_append_option_if_env_set` 把 `SGLANG_ENGINE_CONTEXT_LEN` / `SGLANG_MEM_FRACTION_STATIC` 拼进 SGLang 调用。**env 不设则数组为空,其它脚本(fair_eval / nsys / beam_compare)行为完全不变。**
- `scripts/run_decode_steps_perf_sweep.sh`(本次新增驱动):`export SGLANG_ENGINE_CONTEXT_LEN=6000` / `SGLANG_MEM_FRACTION_STATIC=0.65`,默认开启。
- SGLang 源码 patch 直接改在挂载的 `examples/sglang_beam_search/` checkout 里(PR 分支 `feature/beam_search`),带 `# Local patch (sid-gr-inference)` 标注,可追溯。

### 验证
- 单元级:`verify_sglang_beam_oom_fix.py` 实测 b1024/batch8 SUCCESS(§3.5)。
- sweep 级:完整 108-cell 重跑,**全程 0 失败**,b512/b1024 全部跑通(包括之前不可能的 b1024/batch8)。详见性能对比报告。

---

## 5. 本次改动清单

**新增文件**
- `scripts/run_decode_steps_perf_sweep.sh` —— decode × beam 外层扫描驱动(每对独立 OUT_DIR、失败隔离、默认开启修复配置)
- `tools/summarize_decode_steps_sweep.py` —— 合并 `d{gr}_b{beam}/summary.csv` 成一张总表
- `tools/verify_sglang_beam_oom_fix.py` —— SGLang OOM 复现 / 修复验证脚本

**修改文件**
- `tools/run_sglang_beam_benchmark.py` —— 新增 `--engine-context-len` 参数,透传 `context_length` 给 `sgl.Engine`
- `scripts/run_gr_sglang_perf_sweep.sh` —— env 驱动透传 `SGLANG_ENGINE_CONTEXT_LEN` / `SGLANG_MEM_FRACTION_STATIC`(不设则无影响)
- `scripts/run_decode_steps_perf_sweep.sh`(新建后更新)—— 默认 `SGLANG_ENGINE_CONTEXT_LEN=6000` / `SGLANG_MEM_FRACTION_STATIC=0.65`

**第三方 patch**
- `examples/sglang_beam_search/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:703` —— 放宽 `req_to_token_pool` 槽位硬上限(标注 local patch)

---

## 6. 附录:复现命令

```bash
# 完整 decode × beam 扫描(已带修复配置)
CONTEXT_LENS="1000 2000 5000" BEAM_WIDTHS="256 512 1024" \
BATCH_SIZES="1 2 4 8" REPEAT=3 \
  scripts/run_container.sh scripts/run_decode_steps_perf_sweep.sh

# 单独复现 / 验证 SGLang OOM 修复
scripts/run_container.sh bash -lc \
  "PYTHONPATH=/workspace/sglang_beam_search/python \
   python /workspace/sid-gr-inference/tools/verify_sglang_beam_oom_fix.py"

# 快速冒烟(单 decode / 单 beam)
CONTEXT_LENS="1000 5000" BATCH_SIZES="1 2 4 8" REPEAT=3 RUN_ACCURACY=1 \
  scripts/run_container.sh scripts/quickstart_offline.sh
```

---

*生成时间:2026-07-06。性能数据见同目录 `report_gr_vs_sglang_perf.md`。*
