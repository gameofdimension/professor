# attention-gym 基于 FlexAttention 的 PagedAttention 实现调研

> 更新时间：2026-05-19  
> 调研对象：[meta-pytorch/attention-gym](https://github.com/meta-pytorch/attention-gym)  
> 入口文件：[attn_gym/paged_attention/latency.py](https://github.com/meta-pytorch/attention-gym/blob/29185740237a6e02c55740be8333cb744abccbd7/attn_gym/paged_attention/latency.py)  
> 代码版本：commit `29185740237a6e02c55740be8333cb744abccbd7`

---

## 1. 背景与一句话概括

这个目录下基于 FlexAttention 的 paged attention，核心思想是：

> **把逻辑上"每个 batch 各自一段连续 KV cache"改成"全局物理页池中的离散页"，
> 然后把 FlexAttention 所依赖的 `BlockMask` / `mask_mod` / `score_mod` 从逻辑坐标系映射到物理坐标系。**

它不重写 attention kernel，而是证明了：FlexAttention 的 `BlockMask + mask_mod + score_mod` 可以直接承载分页 KV cache 这种复杂的存储布局。

---

## 2. 目录结构

```
attn_gym/paged_attention/
├── paged_attention.py   # 核心：PagedAttention 类
├── model.py             # NonPagedAttentionLayer / PagedAttentionLayer
├── utils.py             # 页表初始化、mask_mod 生成、block_mask 工具
├── latency.py           # 入口：单层延迟对比 benchmark
└── throughput.py        # 吞吐测试：模拟在线服务，验证最大 batch size
```

---

## 3. 为什么需要 Paged Attention

普通实现中，KV cache 形状为 `[B, H, max_seq_len, D]`，每条请求都按最大序列长度分配连续内存，导致：

- batch 内请求长度不同时存在大量 padding 浪费
- 大 batch / 长上下文时 OOM 风险高

paged attention 借鉴 vLLM 思路：

- 每条序列按固定 `page_size` 切页
- 每个 batch 的逻辑页映射到全局物理页池中的任意一个物理页
- KV cache 统一存储在一块连续的 `[1, H, n_pages * page_size, D]` 张量中
- 避免按 `max_seq_len` 为每个 batch 都预留连续大块空间

`throughput.py` 中给出了数字对比：

- 无 paged attention：以 llama-3.1 的 131072 上下文长度，4GB KV cache 预算下最多服务 **32 条请求**
- 有 paged attention（page_size=128，最多 32768 页）：最多服务 **2448 条请求**，约 **76x** 提升

---

## 4. `PagedAttention` 类：核心数据结构

`paged_attention.py` 中的 `PagedAttention` 类是整个实现的核心。

### 4.1 页表（`page_table`）

```python
self.page_table = -torch.ones((max_batch_size, self.n_pages), dtype=torch.int64, device=device)
```

语义：`page_table[batch, logical_block_idx] = physical_page_idx`

即：某条请求的第几个逻辑块，对应全局页池中哪一个物理页。这是**逻辑地址 → 物理地址**的映射。

### 4.2 容量记录（`capacity`）

```python
self.capacity = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
```

按页粒度记录已分配容量，以 token 数计。注意是"已分配"而非"已写入"——按页增长。

### 4.3 空闲物理页池（`empty_pages`）

```python
self.empty_pages = list(range(n_pages - 1, -1, -1))
```

Python 列表，记录当前可用的物理页索引。通过 `reserve` 分配、`erase` 回收。

### 4.4 物理页到逻辑块的反向映射（`physical_to_logical`）

```python
self.physical_to_logical = -torch.ones((max_batch_size, n_pages), dtype=torch.int64, device=device)
```

语义：`physical_to_logical[batch, physical_page_idx] = logical_block_idx`

这是反向映射，主要供 `get_mask_mod` 和 `get_score_mod` 使用。因为 FlexAttention 在执行时看到的是**物理 KV index**，需要还原成逻辑 KV index 才能正确评估 causal 等语义。

---

## 5. 页的生命周期

### 5.1 分配：`reserve(batch_idx, seq_len)`

```python
def reserve(self, batch_idx, seq_len):
    num_pages_to_allocate = ceil((seq_len - capacity[batch_idx]) / page_size)
    allocated_pages = empty_pages[-num_pages_to_allocate:]
    page_table[batch_idx, start_page_idx:end_page_idx] = allocated_pages
    physical_to_logical[batch_idx, allocated_pages] = arange(start_page_idx, end_page_idx)
    capacity[batch_idx] += num_pages_to_allocate * page_size
```

按需分配，只从空闲页池取必要数量，逻辑连续但物理不要求连续。

### 5.2 回收：`erase(batch_idx)`

找出该 batch 分配过的所有物理页，清空 `page_table` 和 `physical_to_logical`，将物理页放回 `empty_pages`，`capacity` 归零。支持在线服务场景中请求结束后释放资源。

---

## 6. KV 写入：`assign(...)`

这是分页存储真正落地的地方。

```
input_pos ──► logical_block_idx = input_pos // page_size
              logical_block_offset = input_pos % page_size
              physical_block_idx = page_table[batch_idx, logical_block_idx]   ← gather 查表
              addr = physical_block_idx * page_size + logical_block_offset
              k_cache[:, :, addr, :] = k_val
              v_cache[:, :, addr, :] = v_val
```

普通实现直接写 `[batch, pos]`；paged 实现多了一次 `logical pos → logical page + offset → physical page → physical pos` 的两级翻译。

---

## 7. 如何接入 FlexAttention：三层坐标变换

FlexAttention 仍然要求标准的 `(q, k, v, block_mask, score_mod)` 接口。
问题在于：**`k/v cache` 的序列轴已经是"物理页展开后的物理位置"**，所有依赖 KV 下标语义的部件都必须同步转换。

```
用户定义的逻辑语义 (causal / relative bias / ...)
        │
        ▼  convert_logical_block_mask / get_mask_mod / get_score_mod
物理页空间的 BlockMask / mask_mod / score_mod
        │
        ▼  flex_attention(q, k_cache_paged, v_cache_paged, ...)
输出结果（完全等价于逻辑视角下的结果）
```

### 7.1 `convert_logical_block_mask(...)`

将逻辑 block mask 的 `kv_indices`（逻辑块号）通过页表 gather 替换为物理页号：

```python
new_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = torch.gather(
    page_table, 1, block_mask.kv_indices.view(B, -1).to(torch.int64)
).view(block_mask.kv_indices.shape).to(torch.int32)
```

几个重要细节：

1. **列数扩展到 `n_pages`**：逻辑块号最大值为 `MAX_BLOCKS_IN_COL - 1`，物理页号最大值为 `n_pages - 1`（通常大得多）。若不扩展，`_ordered_to_dense` 等内部函数访问时会越界。
2. **同步处理 `full_kv_indices`**：`BlockMask` 除稀疏 `kv_indices` 外还有整块路径 `full_kv_indices`，两者都需要 gather 替换，否则整块路径走到错误物理地址。
3. **用 `BlockMask.from_kv_blocks()` 构造新对象**：不依赖内部 `_replace` API；同时将 `seq_lengths` 的 KV 维度从逻辑空间长度更新为 `n_pages * page_size`（物理空间总长）。
4. **显式校验 `page_size == BLOCK_SIZE`**：`flex_attention` 内部用 `kv_indices * BLOCK_SIZE` 定位物理地址，因此只有 `page_size == BLOCK_SIZE` 时 gather 替换才正确。attention-gym 主动断言这一约束，违反时立即报错。

### 7.2 `get_mask_mod(...)`

包装用户定义的逻辑 `mask_mod`，在每次 token 级 attention 计算前做"物理 KV index → 逻辑 KV index"的翻译：

```python
def new_mask_mod(b, h, q_idx, physical_kv_idx):
    physical_kv_block = physical_kv_idx // page_size
    physical_kv_offset = physical_kv_idx % page_size
    logical_block_idx = physical_to_logical[b, physical_kv_block]   # 反向页表查找
    logical_kv_idx = logical_block_idx * page_size + physical_kv_offset
    return torch.where(
        logical_block_idx >= 0,          # 无效页直接返回 False
        mask_mod(b, h, q_idx, logical_kv_idx),
        False,
    )
```

对于无效物理页（`physical_to_logical` 值为 -1），直接返回 `False`，排除在 attention 计算之外。

### 7.3 `get_score_mod(...)`

与 `get_mask_mod` 逻辑相同，但作用于 score 修改上，无效页返回 `-inf`（softmax 后自然权重为 0）：

```python
def new_score_mod(score, b, h, q_idx, physical_kv_idx):
    ...
    return torch.where(
        logical_block_idx >= 0,
        score_mod(score, b, h, q_idx, logical_kv_idx),
        float("-inf"),
    )
```

这保证了相对位置 bias、head bias 等 score 级修改的逻辑语义不受物理布局影响。

---

## 8. `model.py`：改动极小的 attention 层

对比 `NonPagedAttentionLayer` 与 `PagedAttentionLayer`，两者差异**仅有两点**：

| 对比维度 | NonPagedAttentionLayer | PagedAttentionLayer |
|----------|----------------------|---------------------|
| KV cache 形状 | `[B, H, max_seq_len, D]` | `[1, H, n_pages * page_size, D]`（共享） |
| KV cache 写入 | `k_cache[batch_idx, :, input_pos]` | `paged_attention.assign(batch_idx, input_pos, ...)` |
| block_mask | 逻辑 block_mask | `converted_block_mask` |
| score_mod | 逻辑 score_mod | `converted_score_mod` |

其余逻辑（QKV 投影、RoPE、`flex_attention` 调用、输出投影）完全相同。这体现了该设计的核心价值：

> **把"KV cache 存储方式变化"局部化到缓存写入与 mask/mod 转换层，`flex_attention` 主调用路径几乎不变。**

---

## 9. `utils.py`：辅助工具

### `random_init_paged_attention(...)`

以循环 round-robin 方式为 batch 分配物理页，模拟多请求并行运行时碎片化的页分布，使 benchmark 更接近真实场景。

### `gen_offset(off)`

生成 decode 场景的 causal mask_mod：

```python
def offset(b, h, m, n):
    return m + off[b] >= n
```

`off[b]` 是当前 batch 的序列偏移（decode 时等于已生成的 token 数），用 `m + off[b] >= n` 实现带全局位置偏移的 causal 约束。

### `generate_score_mod(attn_type)`

支持 4 种 score_mod：

- `noop` / `causal`：恒等变换
- `rel`：相对位置 bias `score + (m - n)`
- `head_bias`：per-head bias `score + 2 * h`

覆盖了 paged attention 需要与多种 score 修改机制协同工作的场景。

### `slice_block_mask(...)`

从全局 block mask 中切出某单个请求当前所需的子 mask，供 `throughput.py` 的 prefill 阶段使用。

---

## 10. `latency.py`：入口 benchmark

`latency.py` 不实现算法，而是做**单层 decode latency 对比**：

```
初始化：input_pos（decode 位置）、batch_idx、PagedAttention
        ↓
生成逻辑 block_mask / score_mod
        ↓
转换为物理 converted_block_mask / converted_score_mod
        ↓
分别编译并 benchmark：
  - NonPagedAttentionLayer
  - PagedAttentionLayer
        ↓
输出：两者 latency 及 paged overhead %
```

测试覆盖 `{noop, causal, rel, head_bias}` 四种 attn_type，以及多种 `bsz / max_seq_len / page_size` 组合，验证分页对延迟的额外开销。

---

## 11. `throughput.py`：在线服务模拟

`throughput.py` 模拟了一个 LLM serving server，关注 paged attention 提升可服务 batch size 的能力。

### 服务循环

```
接收请求队列
    │
    ▼ prefill()          ─ 为新请求分配页 + 跑 prompt prefill
    ▼ decode() × gap     ─ 所有活跃 batch 一起解码 gap 步
    ▼ clean()            ─ 完成的请求 erase() 回收页
```

### `prefill_one_token(batch_idx, prompt_len, response_len)`

1. `paged_attention.reserve(batch_idx, prompt_len + response_len)`——提前分配 prompt + response 所需页
2. `slice_block_mask` 切出 prompt 的局部 block mask
3. `convert_logical_block_mask` → `get_score_mod` 转换
4. 跑 `PagedAttentionLayer` 完成 prefill

### `decode()`

- 所有活跃 batch 合并为当前 step 的 batch
- 从全局 causal block mask 中按 `input_pos` 取当前行的 kv_num_blocks / kv_indices
- 转换为物理 block_mask，跑一步 decode

### 吞吐收益

按照 OpenOrca 数据集的 prompt/response 长度分布，4GB KV cache 预算、4 heads、head_dim=64、bfloat16：

| 方案 | 上下文长度 | 最大 batch | 备注 |
|------|-----------|-----------|------|
| 无 paged attention | 131072 | 32 | 每条请求预留完整 max_seq_len |
| 有 paged attention | page_size=128 | ~2448 | 仅分配实际使用的页，约 76x |

---

## 12. 设计三层抽象总结

```
┌─────────────────────────────────────────────────────────┐
│  逻辑层（用户视角）                                        │
│  - batch b 的第 n 个 token                               │
│  - causal / relative bias / head bias 等逻辑语义         │
└──────────────────────┬──────────────────────────────────┘
                       │  PagedAttention（分页映射层）
                       │  - logical token → logical page + offset
                       │  - logical page → physical page（page_table）
                       │  - physical page → logical page（physical_to_logical）
                       │  - convert_logical_block_mask / get_mask_mod / get_score_mod
┌──────────────────────▼──────────────────────────────────┐
│  FlexAttention 执行层                                    │
│  - dense q [B, H, S, D]                                  │
│  - paged-packed k/v [1, H, n_pages*page_size, D]        │
│  - converted block_mask（物理页号）                       │
│  - converted mask_mod / score_mod（物理坐标）             │
└─────────────────────────────────────────────────────────┘
```

---

## 13. 与常规 paged attention 的对比

| 维度 | 本项目 | vLLM 等生产实现 |
|------|--------|----------------|
| 实现层次 | 纯 Python 层，FlexAttention 内核不变 | C++/CUDA 层自定义 kernel |
| KV 地址翻译 | `mask_mod`/`score_mod` 闭包 + `block_mask` gather | kernel 内部直接按页表寻址 |
| 页大小约束 | `page_size == BLOCK_SIZE`（断言强制） | 通常 page_size=16~128，较灵活 |
| 适用场景 | 实验验证、快速原型、FlexAttention 生态 | 生产推理服务 |
| attention 模式扩展性 | 任意 `mask_mod` / `score_mod` 均可组合 | 通常仅支持 causal 等固定模式 |

---

## 14. 参考

- 代码仓库：<https://github.com/meta-pytorch/attention-gym>
- 入口文件：<https://github.com/meta-pytorch/attention-gym/blob/29185740237a6e02c55740be8333cb744abccbd7/attn_gym/paged_attention/latency.py>
- PyTorch FlexAttention 文档：<https://pytorch.org/docs/stable/nn.attention.flex_attention.html>
- vLLM PagedAttention 论文（Kwon et al., 2023）：<https://arxiv.org/abs/2309.06180>
