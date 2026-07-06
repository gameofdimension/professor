# GR vs SGLang 性能对比报告(decode-steps × beam-width 全扫描)

> 模型 Qwen3-1.7B(真实权重)· 硬件 NVIDIA RTX PRO 5000 72GB ×8 · beam-search PR 分支
> `feature/beam_search`。GR = 本项目自研 SID-GR 引擎,SGLang = 上述 PR 分支。两引擎
> 跑同一份确定性 workload、关双方前缀缓存、`temperature=0`。
>
> 数据源:`benchmark_artifacts/sglang_compare/offline_decode_steps_20260706_090925/`
> (`summary.csv` / `summary.md`,共 108 cell)。

---

## 1. 测试配置

| 维度 | 取值 |
|---|---|
| GR decode steps | 2 / 3 / 4(对应 SGLang 3 / 4 / 5,等长 token,见调查报告 §2.7) |
| beam width | 256 / 512 / 1024 |
| context len | 1000 / 2000 / 5000 |
| batch | 1 / 2 / 4 / 8 |
| repeat | 3(取 `wall_ms_median`) |
| 总 cell 数 | 3 × 3 × 3 × 4 = **108** |

SGLang 引擎配置(本次修复后,见调查报告 §4):`context_length=6000`、
`mem_fraction_static=0.65`、放宽 `req_to_token_pool` 的 4096 钳制。**108/108 cell 全跑通,0 失败。**

---

## 2. 核心结论

- **GR 在全部 108 个 cell 胜出**,加速比(SGLang/GR)= **1.52×–3.50×**(均值 2.09×,中位 1.94×)。
- 加速比随 **beam ↑、context ↑、decode-steps ↑** 三维单调放大,且叠加效应明显。
- 峰值:**GR decode=4 / ctx5000 / beam1024 → 3.50×**(batch4:GR 895ms vs SGLang 3134ms)。
- 最差也有 1.52×(最小负载:decode=2 / ctx1000 / beam256)。
- GR 在该负载下的两个结构性优势体现充分:① 紧凑 BeamKV 池使 beam 线性扩展;② 长 context 下 prefill 远快于 SGLang。

---

## 3. 加速比趋势

### 3.1 三维叠加 —— 加速比随 beam × context(对 decode/batch 取均值)

| beam \ ctx | 1000 | 2000 | 5000 |
|---|---|---|---|
| 256 | 1.61× | 1.77× | 1.89× |
| 512 | 1.77× | 2.07× | 2.41× |
| 1024 | 1.89× | 2.38× | **3.07×** |

→ 右下角(大 beam + 长 context)是 GR 优势最大的区域。

### 3.2 峰值区域 —— ctx5000 下,decode × beam

| decode \ beam | 256 | 512 | 1024 |
|---|---|---|---|
| 2 | 1.66× | 2.10× | 2.72× |
| 3 | 1.90× | 2.43× | 3.11× |
| 4 | 2.11× | 2.68× | **3.38×** |

→ decode 越多、beam 越大,GR 越划算。

### 3.3 单维度均值

| 维度 | 加速比均值(范围) |
|---|---|
| beam 256 / 512 / 1024 | 1.76× / 2.08× / 2.45× |
| context 1000 / 2000 / 5000 | 1.76× / 2.07× / 2.45× |
| decode 2 / 3 / 4 | 1.94× / 2.11× / 2.24× |

### 3.4 Top / Bottom cell

| | cell | 加速比 | GR ms | SGLang ms |
|---|---|---|---|---|
| **Top** | d4 / ctx5000 / b1024 / batch4 | **3.50×** | 895 | 3134 |
| | d4 / ctx5000 / b1024 / batch8 | 3.48× | 1781 | 6197 |
| | d4 / ctx5000 / b1024 / batch2 | 3.39× | 465 | 1579 |
| **Bottom** | d2 / ctx1000 / b256 / batch2 | 1.52× | 73 | 112 |
| | d2 / ctx1000 / b256 / batch1 | 1.54× | 44 | 68 |

---

## 4. GR 内部分解

### 4.1 prefill 主导长 context(GR prefill 占总耗时比例)

| context | prefill 占 GR wall |
|---|---|
| 1000 | 30% |
| 2000 | 42% |
| 5000 | **62%** |

长 context 下 prefill 是 GR 的主要耗时,而 GR prefill 又远快于 SGLang —— 这是加速比随 context 放大的主因。

### 4.2 GR 随 beam 线性扩展(ctx1000 / decode=2,GR wall ms)

| beam | batch1 | batch2 | batch4 | batch8 |
|---|---|---|---|---|
| 256 | 44 | 73 | 135 | 248 |
| 512 | 61 | 104 | 184 | 357 |
| 1024 | 92 | 153 | 293 | 579 |

beam 256→1024(4×),GR wall 约 2.0–2.1× —— 平滑亚线性,无爆点。SGLang 在大 beam 下
既慢又(修复前)OOM,差距随 beam 拉开。

---

## 5. 为什么 SGLang 落后(简述)

详见 [`report_test_framework_and_sglang_oom.md`](report_test_framework_and_sglang_oom.md)。两点结构性原因:

1. **大 beam 显存模型低效**:SGLang beam search 为 `batch × beam` 条序列持有全词表(151936)
   logits 张量,`[batch×beam, vocab]` 随 beam 线性膨胀(b1024/batch8 ≈ 5GB),既慢又吃显存。
   原始 SGLang 在 `batch×beam > 4096` 直接 OOM(4096 硬钳制),本次修复后才跑通。
2. **长 context prefill 慢**:GR 用预分配紧凑 BeamKV/ContextKV 池 + `gr-decode_atten` 真 kernel
   + CUDA graph replay,prefill 显著快于 SGLang;context 越长差距越大。
3. GR 是 **item-constrained decode**(只算候选小词表),根本不产生全词表 logits,所以 beam 越大优势越大。

---

## 6. 结论

在 SID-GR 的目标负载(长 context + 短 decode + 大 beam)全矩阵上:

- **GR 一致地快于 SGLang 1.52×–3.50×**,且优势随 beam / context / decode-steps 三维放大;
- 峰值出现在 **大 beam(1024)+ 长 context(5000)+ 多 decode(4)** 组合,达 **3.5×**;
- 即便最轻负载也保持 1.5× 以上;
- GR 的 beam 扩展性平滑线性,SGLang 在大 beam 下需要额外修复才能跑通且仍慢。

这验证了本项目自研引擎在该细分负载下相对最强开源基线(SGLang beam-search PR)的速度优势。

---

## 7. 复现

```bash
CONTEXT_LENS="1000 2000 5000" BEAM_WIDTHS="256 512 1024" \
BATCH_SIZES="1 2 4 8" REPEAT=3 \
  scripts/run_container.sh scripts/run_decode_steps_perf_sweep.sh
# 产物:benchmark_artifacts/sglang_compare/offline_decode_steps_<RUN_ID>/{summary.md,summary.csv}
```

---

*生成时间:2026-07-06。测试框架与 OOM 调查看同目录 `report_test_framework_and_sglang_oom.md`。*
