# DeepEP API 与版本演化梳理

> 基于 [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) 公开仓库的源码与 git
> 历史整理(核对范围: main / v2.5 / hybrid-ep 分支)。所有 commit 均可在仓库中
> `git show <hash>` 复核,README 引用为原文。

## 一、总体时间线与分叉图

hybrid-ep **确实从 main 主线分叉而来**: 分叉点是两线的共同祖先
`abba6ad`(2025-09-17, "Fix hidden_size % 128 != 0 in intranode kernels" #413)——
分叉时仓库尚为纯 V1 形态(原生 `deep_ep/buffer.py` 存在,`hybrid_ep_buffer.py` 不存在)。
此后两线并行演进、无任何合流(各自 38 / 41 个独有 commit)。

```
2025 初 ── V1 发布: Buffer(normal + low-latency 双模式), 传输 NVSHMEM
   │
   ▼  main 线继续 V1 演进(约 7 个月)
abba6ad ══╦═════════════════════════════════════════════════════════════
2025-09-17 ║  ← 分叉点(共同祖先, V1 形态)
           ║
           ╠─► hybrid-ep 线(独立 38 commit, 线端 4214430)
           ║      新增 HybridEPBuffer(分叉点上不存在此文件)
           ║      节点内 TMA / 节点间 DOCA verbs —— 摆脱 NVSHMEM
           ║      保留原生 Buffer, 未经历 EPv2 重构
           ║      ┈┈ 4214430(2026-05-20)"Handle token padding & inter-node check"
           ║
           ╚─► main 线(独立 41 commit)
                  b306af0(2026-04-30)V2 公开发布 "Public release 26/04":
                      ElasticBuffer 引入; Buffer 降级为 buffers/legacy.py 兼容层
                      DeepJIT 运行时编译; NCCL Gin(GPU 直发, 复用 NCCL comm)
                  ┈┈ V2.x 迭代: Engram / PP / Bucket collectives(实验)
                  V2.5(当前):
                      ElasticBuffer 拆分为 EPBuffer / EngramBuffer / PPBuffer / BucketBuffer
                      彻底移除 V1(legacy Buffer)与 NVSHMEM 依赖
```

## 二、三个世代的本质差异

| 维度 | V1(`Buffer`) | hybrid-ep(`HybridEPBuffer`) | V2 世代(`ElasticBuffer` → `EPBuffer` 系) |
|---|---|---|---|
| 定位 | MoE 训练 + 推理 prefill(normal)/decode(low-latency)双模式 | 面向大规模推理的融合 EP(后也被用于训练) | EP 为核心、扩展为全家桶(PP/CP/DP/远程内存) |
| 节点内传输 | NVSHMEM over NVLink | **TMA 指令**(SM 占用极小,支持非 NVLink 的 PCIe 互联) | NCCL Gin(设备侧轻量 API) |
| 节点间传输 | NVSHMEM IBGDA(宿主机驱动需开 NVreg 参数, 见附录 B) | DOCA gpunetio/ibverbs 直连(自带 doca objects,无 NVSHMEM 依赖) | NCCL Gin(要求较新 NCCL, 见 README 要求) |
| 内核交付 | 预编译进扩展 | 预编译 + JIT backend(`deep_ep/backend/`) | **DeepJIT 全运行时编译**(装包无需 GPU/arch) |
| Buffer 模型 | NVL+RDMA 双域一次性预分配 | 惰性分配、按需扩 | 弹性增长 → V2.5 起按用途拆四类 + BufferAllocator 规划 |
| 外部依赖 | NVSHMEM(独立库) | rdma-core + DOCA objects 打包 | 只依赖 NCCL/CUDA(**V2.5 起彻底去掉 NVSHMEM**) |

## 三、分支拓扑(关键且容易被误解的一点)

- **hybrid-ep 从 main 分叉**, 分叉点 `abba6ad`(2025-09-17), 早于 EPv2 重构约 7 个月
  —— 这解释了它为何保留原生 `Buffer`
- **两线此后从未合流**(双向 `git merge-base --is-ancestor` 均为否, 独有 commit 38 vs 41):
  - `HybridEPBuffer` 是分叉后 hybrid-ep 线独有新增(分叉点上不存在), main 任何时点都没有它
  - `ElasticBuffer`/`EPBuffer` 只存在于 main, hybrid-ep 上没有
- hybrid-ep 线的构建因此是**唯一同时提供 `Buffer` + `HybridEPBuffer` 的组合**
- main 上 V1 的死法分两步:
  1. V2.0(`b306af0`):`deep_ep/buffer.py` 被删除,`Buffer` 搬入 `deep_ep/buffers/legacy.py`
     降级为兼容层(此时与 `ElasticBuffer` 并存)
  2. V2.5:`from .buffers.legacy import Buffer` 导出被移除,连同 NVSHMEM 依赖彻底删除

### 各时点 `deep_ep/__init__.py` 导出面(代码快照证据)

| 时点 | 相关导出 |
|---|---|
| hybrid-ep 线(例 4214430) | `Buffer`(buffer.py)+ `HybridEPBuffer`(hybrid_ep_buffer.py) |
| V2.0 发布 `b306af0` | `from .buffers.legacy import Buffer` + `from .buffers.elastic import ElasticBuffer, EPHandle` |
| V2.5 前 main `d4f41e4`(2026-05-26) | 同上(双 API 并存到最后一刻) |
| 当前 `origin/main`(V2.5+) | 仅 `EPBuffer/EPHandle`、`EngramBuffer`、`PPBuffer`、`BucketBuffer`、`BufferAllocator`、`BufferBase` |

## 四、能力版图的扩张轨迹

1. **V1**: 只有 expert dispatch/combine(含 fp8 dispatch、low-latency 模式)
2. **V2**: JIT 化 + Gin 传输 + 更大 scale 域;实验性引入 Engram(GPU/CPU 远程内存)、PP p2p、
   Bucket collectives(all-gather/reduce-scatter/all-reduce —— **"DeepEP 开始碰 CP/DP 通信"的起点**,
   仅限 main 新 API,老 `Buffer` 无此能力)
3. **V2.5**: Bucket collectives 面向 CP/DP 正式化;新增动态冗余专家原语
   (`lb_prefetch_weights`/`lb_reduce_grads`,MoonEP/UltraEP 思路);deferred epilogue、
   cached layout 等;另有 Ascend 移植版(DeepEP-Ascend)

## 附录 A: 证据锚点(均可复核)

| 结论 | 复核方法 |
|---|---|
| ElasticBuffer 引入/移除 | `git log -S "class ElasticBuffer" --all` → 引入于 `b306af0`(EPv2 公开发布), 移除于 V2.5(`1a1ffba`/`def8651`) |
| V1 `buffer.py` 删除点 | `git log --diff-filter=D -- deep_ep/buffer.py` → `b306af0`(删除后经 `buffers/legacy.py` 保留, V2.5 连导出一并移除) |
| 分支归属 | `git branch -r --contains 4214430` → 仅 `origin/hybrid-ep`;`git branch -r --contains b306af0` → main 系 |
| **分叉点** | `git merge-base origin/hybrid-ep origin/main` → `abba6ad`(2025-09-17);该点上 `deep_ep/buffer.py` 存在、`deep_ep/hybrid_ep_buffer.py` 不存在 |
| 双线未合流 | `git merge-base --is-ancestor` 双向均为否;独有 commit 数 `git rev-list --count origin/main..origin/hybrid-ep` = 38 / 反向 = 41 |
| V2.5 官方说明 | README 原文: "Split `ElasticBuffer` into `EPBuffer`, `EngramBuffer`, `PPBuffer`, and `BucketBuffer`"; "Fully remove V1, including its APIs, NVSHMEM backend" |

## 附录 B: V1(`Buffer`)节点间通信的宿主机前置 —— IBGDA 驱动参数

V1 的节点间路径(NVSHMEM IBGDA)要求宿主机 NVIDIA 驱动开启以下模块参数,
与 [vLLM ep_kernels 部署文档](https://github.com/vllm-project/vllm/tree/main/tools/ep_kernels)
中 `configure_system_drivers.sh` 所做的一致:

```
options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"
```

- **检查**: `grep -i streammem /proc/driver/nvidia/params`(未配置时为 `EnableStreamMemOPs: 0`)
- **缺失时的典型症状**: 首次构建 `Buffer` 挂死于 NVSHMEM 初始化的 team barrier
  (调用栈形如 `nvshmemi_team_init_nccl_comm → nvshmemi_barrier → cudaStreamSynchronize`),
  并伴随 WARN: `NULL value cannot allocate mlx5dv_devx_uar` /
  `ibgda_alloc_and_map_qp_uar with GPU as handler failed. We may need to enter the CPU fallback path.`
  —— 根因是 IBGDA 无法把 QP 的 UAR(敲门铃寄存器)以 GPU 句柄映射进 GPU 地址空间,
  属宿主机驱动配置缺失,与 DeepEP/NVSHMEM 软件版本无关
- **修复**: 宿主机写入上述 modprobe 配置 + 更新 initramfs(`update-initramfs -u` 或
  `dracut --force`)+ **重启**;重启后 `EnableStreamMemOPs: 1`
- **注意**: hybrid-ep 线与 V2 世代(Gin/DOCA 路径)不依赖 NVSHMEM IBGDA,不受该参数影响
