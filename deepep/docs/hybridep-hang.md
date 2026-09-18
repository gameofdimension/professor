# HybridEP dispatch 死锁问题报告(2026-09-16 定位,09-17 机制定案)

## 1. 问题现象

**场景**: GLM-5.3-Flash-mini(288 路由专家 MoE)在 2×8 H20 上以
`dispatcher: hybridep` 训练,`EP_SIZE=16` → 每 rank 18 个本地专家。
> 符号约定: 下文 **E = `num_local_experts` = 每 rank 分到的本地专家数 =
> 总专家数 ÷ EP_size**。本文所有出现 E 的地方均指此量(如 E=18 即
> 288 专家在 16 rank 上各得 18 个)。

**表现**:

- 训练在 **step 0 的第一次 MoE dispatch** 卡死: 进度条停在 0 步,首个 MoE
  层 forward 无返回;
- 两节点 **16 个 rank 全部 GPU 利用率 100%**(kernel 级自旋忙等,不是空闲
  死锁),显存仅模型占用(~11.5GB/卡);
- **完全静默**: 无报错、无 traceback、无超时打印、无 NCCL 错误,只能靠
  外部 timeout 击杀(torchrun exit=124);
- py-spy 栈: 全部 rank 停在 `dispatch_with_permute`
  (`deep_ep/hybrid_ep_buffer.py:370`)→ `metadata_preprocess_core` →
  `pad_tokens_per_expert_kernel` 的 `cuLaunchKernel`(同 stream 前驱
  kernel 永不完成),native 栈即 JIT scan kernel;
- **单机 8 卡同样复现**(与跨机/网络无关)。

**触发边界**(对等测试矩阵实测,单机/跨机一致):

| num_local_experts | 1 | 2 | 12 | 16 | 17 | 18 | 20 | 32 | 36 |
|---|---|---|---|---|---|---|---|---|---|
| 缺省配置结果 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

与 token 数(5~2048)、hidden(256/4096)、top-k(2/8)、NIC 拓扑、节点数
均无关;唯一开关是 scan kernel 的 block 数(见 §3)。

**生产影响面**: 官方 9×8=EP72 拓扑为 4 专家/rank,不受影响;2×8 冒烟
(18/rank)必中;当前 deep_ep 版本下 hybridep 要求 EP_size ≥ 18(288 专家),
即至少 3×8 节点。

## 2. DeepEP 版本

| 项 | 值 |
|---|---|
| 仓库/分支 | deepseek-ai/DeepEP, **hybrid-ep** 分支 |
| commit | `42144303752422ade37f24bca9e2dde12df70e09`(PR #638) |
| 版本号 | `1.2.1+4214430` |
| 构建补丁 | cuda13 小补丁(CCCL include + pynvml 改名),不改 kernel 源码 |
| bug 引入时间 | hybrid-ep 分支首个提交 #420 起(非回归) |
| 升级能否修复 | **否**——至最新 tip,scan kernel 文件(`hybrid_ep_backend.cuh`)一字未改 |

## 3. 原因(完整逻辑链条)

### 环境事实

1. **H20 = 78 个 SM**(实测 `torch.cuda.get_device_properties`;
   TechPowerUp/WCCFTech 交叉一致;NVIDIA 无公开 H20 datasheet);
   Hopper 每 SM 寄存器文件 **65,536** 个 32 位寄存器;
2. deep_ep scan(metadata 预处理)kernel 的 grid 缺省 **NUM_OF_BLOCKS=108**
   (`config.cuh` 中 `value_or(108)` **硬编码**,不随设备 SM 数调整);
   **108 > 78**;
3. 每 block **256 线程**(deep_ep 配置,`NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API`
   缺省 256,非硬件线;硬件上限 1024/block)。

### 因果链(每环均已实证)

```
① E > 16   (E = num_local_experts, 每 rank 本地专家数, 见 §1 符号约定)
      ↓ (kernel 源码含长度为 E 的每线程数组:
         token_local_experts_routing_map_sum[E] 等;
         ptxas 全部放寄存器, 实测 0 spill)
② 每线程寄存器需求增大
      ↓ (ptxas -v 实测曲线: E=8→91, E=16→127, E=17→166, E=18→146,
         E=20→157, E=32→213; 复刻 JIT 的 nvcc 命令以 -Xptxas -v 编译
         获得, 方法见 §5)
③ 每 block 占用 256×R 个寄存器, 越过 65536 的一半
      ↓ (R = 每线程寄存器数; 2 blocks/SM 要求 2×256×R ≤ 65536 ⟺ R ≤ 128)
④ 占用度从 2 blocks/SM 掉到 1 block/SM
      ↓
⑤ 常驻上限 = 78 SM × 1 = 78 < grid 108 → 30 个 block 滞留发射队列
      ↓ (E≤16 时 R≤128 → 78×2=156 ≥ 108, 全部常驻, 故不触发)
⑥ scan kernel 的跨块 flag 协议: 每 block 写完自己的部分和槽后,
   自旋等待全部其他 block 的 PRIV_SUM 标志 —— 汇合点要求所有 block
   同时在执行(软件栅栏的隐形契约, 而 __launch_bounds__(256,1) 只向
   编译器要了 1 block/SM 的驻留保障)
      ↓
⑦ 滞留 block 的 flag 永不写出 → 常驻 block 永不结束 → SM 永不空闲
   → 滞留 block 永不进驻(循环等待闭合)
      ↓
⑧ 死锁: GPU 100% 自旋、静默(scan 的 flag 等待无看门狗; 同库 custom
   allgather kernel 反而有 clock64 超时打印)
```

**边界恰为 16 的原因**: E=16 编出 **127** 个寄存器,离 128 的悬崖只差 1。

**上游为何从未发现**: H100/H800(132 SM)、B200 等 ≥108 SM 的卡,即使
占用度 1 block/SM 也够 108 个 block 全部常驻——**H20(78 SM)特有暴露面**。

## 4. 解决方法(附参数含义)

### 方案 A(已采用): 调小 scan kernel 的 block 数 → `num_sms_preprocessing_api=24`

- **参数**: `HybridEPBuffer` 构造参数 `num_sms_preprocessing_api`,经
  Configurer 变为 `num_of_blocks_preprocessing_api`,即 scan kernel 的
  JIT 模板常量 `NUM_OF_BLOCKS`(grid 大小);命名中的 "sms" 是"SM 预算"
  的设计意图(dispatch/combine 各有同族参数)。
- **效果**: `scan<<<108,256>>>` → `scan<<<24,256>>>`;24 ≤ 78,最坏占用度
  (1 block/SM)下也全部常驻,汇合点必然可达。**不修寄存器悬崖,只消除
  "grid 超出常驻能力"这一环**。
- **正确性**: block 数只改变并行切分粒度,不改变归约数学;1/8/24/32 实测
  全过,训练 loss 与 torch dispatcher 路径交叉吻合。
- **性能**: metadata 扫描微秒级,无代价。
- **本仓库落地**: `fused_a2a.py` 的 `init_hybrid_ep_buffer` 缺省传 24
  (commit `b1a2d70f`),环境变量 `HYBRIDEP_NUM_SMS_PREPROCESSING` 可调。
- **取值约束**: 须 ≤ 单卡 SM 数(78);建议与 dispatch/combine 的 24 一致。

### 方案 B(应急替代): `DISPATCHER=torch`

- **参数**: NeMo `BackendConfig.experts`/dispatcher 选择(hybridep|torch);
  torch 为不依赖 deep_ep 的保守路径。
- **代价**: 放弃 hybridep 通信性能(本负载实测 step1 tps 6122 vs hybridep
  16594,约 2.7×)。

### 方案 C(上游修复方向,需改 deep_ep 源码)

- 构造时按设备 SM 数与 num_local_experts 对 `NUM_OF_BLOCKS` 做
  clamp/fail-fast 校验(库内已有 `Invalid BufferConfig` 的先例);
- 或 `__launch_bounds__(256, 2)` 强制双 block 常留(编译器被迫压寄存器,
  可能引入 spill);
- 或给 scan 的 flag 等待加 clock64 看门狗(对齐 custom allgather 的做法),
  把静默死锁变成可诊断报错;
- 或协议改造,不依赖全 grid 常驻。

### 相关参数一览

| 参数 | 层 | 缺省 | 含义 / 与本问题的关系 |
|---|---|---|---|
| `num_sms_preprocessing_api` | Python ctor | 108(硬编码) | scan kernel grid 数;**本问题主开关** |
| `HYBRIDEP_NUM_SMS_PREPROCESSING` | NeMo env | 24 | 上述参数的本仓库覆盖入口 |
| `num_sms_dispatch_api` / `num_sms_combine_api` | Python ctor | 24 | dispatch/combine kernel 的 SM 预算 |
| `NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API` | env | 256 | 每 block 线程数;决定寄存器悬崖位置(R 临界 = 65536/(2×线程数));理论上是另一规避旋钮但改变 warp 归约行为,不推荐 |
| `NUM_OF_TOKENS_PER_CHUNK_*_API` | env | 64 | 各 kernel 的 token 分块;与本问题无关 |
| `HYBRID_EP_ENABLE_MANUAL_NIC_MAPPING` + `HYBRID_EP_NIC_MAPPING` | env | off | bond 网卡手动映射,规避多机 ctor 的 select_net 段错误——**另一独立问题**,多机必需 |
| (token 数对齐) | API 契约 | — | dispatch 的 token 数须 4 对齐(16B),否则触发**又一个独立死锁源**;NeMo padding 已掩盖 |

## 5. 最小复现

零 NeMo 依赖,直接调用 `HybridEPBuffer.dispatch_with_permute`。以下即完整
复现脚本(仓库同款: `lab/jobs/hybrid_dispatch_repro.py`),单机 8 卡约 30s
出结果:

```python
#!/usr/bin/env python3
"""hybrid_dispatch_repro.py — env: NUM_EXPERTS(须被 world 整除)
TOKENS(保持 4 的倍数) TOPK HIDDEN SMS_PREPROC(0=缺省 108)"""
import faulthandler
import os

faulthandler.enable()
import torch
import torch.distributed as dist
from deep_ep import HybridEPBuffer

E_TOTAL = int(os.environ.get("NUM_EXPERTS", "16"))
TOKENS = int(os.environ.get("TOKENS", "8"))
TOPK = int(os.environ.get("TOPK", "2"))
HIDDEN = int(os.environ.get("HIDDEN", "256"))
SMS = int(os.environ.get("SMS_PREPROC", "0"))

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
group = dist.new_group(ranks=list(range(world)))
e = E_TOTAL // world

buf = HybridEPBuffer(
    group=group, hidden_dim=HIDDEN, max_num_of_tokens_per_rank=TOKENS,
    num_local_experts=e, use_fp8=False, num_sms_dispatch_api=24,
    num_sms_combine_api=24,
    **({} if SMS <= 0 else {"num_sms_preprocessing_api": SMS}))

torch.manual_seed(1234 + rank)
hidden = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
routing = torch.zeros(TOKENS, E_TOTAL, dtype=torch.bool, device="cuda")
idx = torch.stack([torch.randperm(E_TOTAL, device="cuda")[:TOPK] for _ in range(TOKENS)])
routing.scatter_(1, idx, True)
probs = torch.rand(TOKENS, E_TOTAL, device="cuda") * routing
probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-9)

print(f"[rank {rank}] dispatch enter", flush=True)
out, out_probs, _, _, handle = buf.dispatch_with_permute(
    hidden=hidden, routing_map=routing, probs=probs, scaling_factor=None,
    num_of_experts_per_rank=e, num_permuted_tokens=None, non_blocking=False)
print(f"[rank {rank}] dispatch OK", flush=True)

combined, _ = buf.combine_with_unpermute(hidden=out, probs=out_probs, handle=handle)
dist.barrier()
print(f"[rank {rank}] PASS", flush=True)
```

运行(三条分别对应卡死 / 对照 / 佐证):

```bash
# 卡死: E=36/rank, 缺省 108 blocks —— 停在 "dispatch enter", GPU 100%, exit=124
NUM_EXPERTS=288 TOKENS=8 torchrun --standalone --nproc_per_node=8 hybrid_dispatch_repro.py
# 对照: E=16/rank —— 全 rank PASS
NUM_EXPERTS=128 TOKENS=8 torchrun --standalone --nproc_per_node=8 hybrid_dispatch_repro.py
# 佐证: E=36/rank + 32 blocks —— PASS, 证明开关是 block 数
NUM_EXPERTS=288 TOKENS=8 SMS_PREPROC=32 torchrun --standalone --nproc_per_node=8 hybrid_dispatch_repro.py
```

双节点: 两节点分别以 `torchrun --nnodes=2 --nproc-per-node=8
--node-rank=<0|1> --master-addr=<worker-0 地址>` 运行同一脚本,并携带 §4
参数表中的 NIC 映射两个环境变量(bond 网卡环境构造期需要)。

**寄存器曲线验证**(纯编译,不占 GPU、不跑 kernel): 以上脚本使用的配置
经 Configurer 生成 JIT 模板实例;手工生成等价的实例化文件并按 JIT 的
nvcc 命令编译(`-arch=sm_90 -O3 --expt-relaxed-constexpr
-DHYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE -I<deep_ep 头文件目录>
-Xptxas -v`),即可在输出中读到 scan kernel 各实例化的 `Used N registers`
——§3 的 91/127/146~213 曲线即由此获得(仓库封装好的工具:
`lab/jobs/ptxas_reg_check.sh`)。
