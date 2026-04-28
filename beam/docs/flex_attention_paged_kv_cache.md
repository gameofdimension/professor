# 用 FlexAttention 实现 PagedAttention 语义

> 更新时间：2026-04-27  
> 研究背景：探索在不修改 FlexAttention 算子本身的前提下，仅通过 Python 层的
> `mask_mod` / `score_mod` 闭包与页表映射，实现与 PagedAttention 等价的分页
> KV Cache 访问语义。

---

## 核心思路

PagedAttention 的本质是：**把连续的逻辑 KV 序列打散存储到不连续的物理内存页**，
从而消除碎片、支持跨请求共享 KV Cache。

FlexAttention 提供的 `mask_mod(b, h, q_idx, kv_idx)` 回调在**每次计算 attention
score 前**被调用，`kv_idx` 指向物理 KV 张量的实际偏移——这正好可以在闭包里做
"物理地址 → 逻辑地址" 的翻译，从而把 causal mask 等原始语义映射到物理布局上。

```
逻辑视角                    物理视角（KV Cache）
─────────────────           ───────────────────────────────────
seq[0]  seq[1]              page0 [tok0, tok1]   ← batch-0 逻辑块0
seq[2]  seq[3]    页表→      page2 [tok2, tok3]   ← batch-0 逻辑块1
seq[4]  seq[5]              page4 [tok4, tok5]   ← batch-0 逻辑块2
                            page1 [tok0, tok1]   ← batch-1 逻辑块0
                            page3 [tok2, tok3]   ← batch-1 逻辑块1
```

三步完成映射，**算子代码零改动**：

1. **写入**：prefill 时按页表散写物理 KV Cache。  
2. **mask_mod 闭包**：decode/prefill attention 时，把物理 `kv_idx` 翻译回逻辑位置，再套 causal + valid-length 条件。  
3. **block_mask 修补**：`create_block_mask` 生成的块索引是逻辑块号，用一次 `torch.gather` 把它替换成物理页号。

---

## 完整可运行示例

```python
"""
用 FlexAttention 实现 PagedAttention 语义（最小完备示例）
依赖：torch >= 2.5

场景：
  - batch=2（两个请求）；序列长 6 和 4
  - page_size=2（每页存 2 个 token）
  - num_heads=1, head_dim=8
  - 物理页池共 6 页；两个请求从同一池中分配，页不连续
"""
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# ─── 超参 ──────────────────────────────────────────────────────────────────
B, H, D   = 2, 1, 8
PAGE_SIZE = 2
N_PAGES   = 6          # 物理页总数

# ─── 1. 物理 KV-cache：所有 batch 共享一块连续张量 ──────────────────────────
#   shape: (1, H, N_PAGES * PAGE_SIZE, D)
k_cache = torch.randn(1, H, N_PAGES * PAGE_SIZE, D)
v_cache = torch.randn(1, H, N_PAGES * PAGE_SIZE, D)

# ─── 2. 页表：逻辑块号 → 物理页号 ────────────────────────────────────────────
#   page_table[b, logic_block] = physical_page_idx  （-1 表示未分配）
page_table = torch.tensor([
    [0, 2, 4, -1],   # batch-0：逻辑块0→物理页0，块1→页2，块2→页4
    [1, 3, -1, -1],  # batch-1：逻辑块0→物理页1，块1→页3
], dtype=torch.long)

# 反向页表：physical_page → logic_block（供 mask_mod 使用）
physical_to_logical = torch.full((B, N_PAGES), -1, dtype=torch.long)
for b in range(B):
    for logic, phys in enumerate(page_table[b]):
        if phys >= 0:
            physical_to_logical[b, phys] = logic

# ─── 3. Prefill：把 KV 按页表散写到物理 cache ────────────────────────────────
seq_lens = [6, 4]

def write_kv(b, seq_len, k_val, v_val):
    for pos in range(seq_len):
        phys_addr = page_table[b, pos // PAGE_SIZE].item() * PAGE_SIZE + pos % PAGE_SIZE
        k_cache[0, :, phys_addr, :] = k_val[pos]
        v_cache[0, :, phys_addr, :] = v_val[pos]

for b in range(B):
    write_kv(b, seq_lens[b],
             torch.randn(seq_lens[b], H, D),
             torch.randn(seq_lens[b], H, D))

# ─── 4. mask_mod：物理 kv_idx → 逻辑位置 → causal + valid ────────────────────
def make_paged_mask_mod(page_table, physical_to_logical, page_size, seq_lens_t):
    def mask_mod(b, h, q_idx, physical_kv_idx):
        phys_page    = physical_kv_idx // page_size
        page_offset  = physical_kv_idx %  page_size
        logic_block  = physical_to_logical[b, phys_page]
        logic_kv_idx = logic_block * page_size + page_offset
        return (logic_block >= 0) & (logic_kv_idx <= q_idx) & (logic_kv_idx < seq_lens_t[b])
    return mask_mod

seq_lens_t = torch.tensor(seq_lens)
mask_mod   = make_paged_mask_mod(page_table, physical_to_logical, PAGE_SIZE, seq_lens_t)

# ─── 5. 构造 block_mask，把逻辑块号替换为物理页号 ──────────────────────────────
Q_LEN = max(seq_lens)

block_mask = create_block_mask(
    mask_mod,
    B=B, H=None,
    Q_LEN=Q_LEN,
    KV_LEN=N_PAGES * PAGE_SIZE,   # 物理 KV 空间总长
    device="cpu",
)

# kv_indices 当前是逻辑块号，用 gather 换成物理页号
kv_idx          = block_mask.kv_indices                      # (B, H, Qb, Kb)
B_, H_, Qb, Kb  = kv_idx.shape
kv_idx_flat     = kv_idx.view(B_, -1)
phys_kv_idx     = torch.gather(
    page_table[:, :kv_idx_flat.max().item() + 1],
    1,
    kv_idx_flat.clamp(min=0),
).view(B_, H_, Qb, Kb)

block_mask = block_mask._replace(kv_indices=phys_kv_idx)

# ─── 6. 调用 FlexAttention（算子代码完全不变）────────────────────────────────
query = torch.randn(B, H, Q_LEN, D)

output = flex_attention(
    query,
    k_cache.expand(B, -1, -1, -1),
    v_cache.expand(B, -1, -1, -1),
    block_mask=block_mask,
)

print("output shape:", output.shape)   # → (2, 1, 6, 8)
print("OK — FlexAttention 内核零改动，PagedAttention 语义在 Python 层完成。")
```

---

## 关键步骤说明

| 步骤 | 位置 | 作用 |
|------|------|------|
| 物理 KV-cache | `k_cache shape=(1,H,N_PAGES*PAGE_SIZE,D)` | 所有 batch 共享，页不需要连续 |
| 散写（prefill） | `write_kv()` | 唯一一次逻辑→物理翻译，发生在写阶段 |
| `mask_mod` 闭包 | `make_paged_mask_mod()` | 运行时把物理 `kv_idx` 翻译回逻辑位置，套 causal + 有效长度条件 |
| block_mask 修补 | `torch.gather(page_table, ...)` | 一行代码把块索引从逻辑空间换到物理空间 |
| 调用 | `flex_attention(q, k_cache, v_cache, block_mask=...)` | 算子本身完全不变 |

---

## 为什么必须替换 `block_mask` 的 `kv_indices`

### `create_block_mask` 在哪个地址空间工作

`create_block_mask` 内部把 KV 序列**均匀切块**（默认 block size 128），然后对每个
`(query block, kv block)` 对调用 `mask_mod` 采样，判断该对块是否整体有效。
它输出的 `kv_indices[b, h, q_block, :]` 保存的是"该 query block 需要关注的 KV
**块号**列表"——这个块号是在**被调用时传入的 `KV_LEN` 所定义的连续空间**内的
顺序编号（第 0 块、第 1 块……）。

在本方案中我们传入 `KV_LEN = N_PAGES * PAGE_SIZE`，因此 `create_block_mask`
将整个物理 KV 空间视作一段**从 0 开始连续排列**的地址，输出的块号也是
从 0 开始的**连续块序号**。

### 问题：连续块序号 ≠ 物理页号

`flex_attention` 运行时会直接把 `kv_indices` 中的值乘以 `BLOCK_SIZE`，当作物理
KV 张量上的起始偏移来加载数据：

```
# flex_attention 内部等效行为
block_start = kv_indices[b, h, q_block, k] * BLOCK_SIZE
load kv_cache[b, h, block_start : block_start + BLOCK_SIZE, :]
```

但在 PagedAttention 中，"逻辑块 1"的数据可能实际存储在物理页 4，
"逻辑块 2"的数据存储在物理页 7——**顺序块序号 ≠ 物理页号**。
若不替换，`flex_attention` 会去错误的物理位置取数据。

### 具体示例

以 `page_size=2`、batch-0 的页表 `[0, 2, 4]` 为例：

| 逻辑块 | create_block_mask 输出的块号 | 实际物理页号 |
|--------|------------------------------|--------------|
| 0      | 0                            | 0 ✓ (恰好相同) |
| 1      | 1                            | 2 ✗ (差 1 页) |
| 2      | 2                            | 4 ✗ (差 2 页) |

若不替换，块号 1 会被当作物理页 1（属于 batch-1），块号 2 会被当作物理页 2
（batch-0 的逻辑块 1），计算结果完全错误。

### 修复：用页表做一次 gather

```python
# kv_indices 当前是连续块序号（逻辑块号）
kv_idx_flat = block_mask.kv_indices.view(B_, -1)

# 用页表把逻辑块号映射到物理页号
phys_kv_idx = torch.gather(
    page_table[:, :kv_idx_flat.max().item() + 1],
    1,
    kv_idx_flat.clamp(min=0),
).view(B_, H_, Qb, Kb)

block_mask = block_mask._replace(kv_indices=phys_kv_idx)
```

`torch.gather` 的语义是 `phys_kv_idx[b, i] = page_table[b, kv_idx_flat[b, i]]`，
即把每个逻辑块号查表换成对应的物理页号。替换后，`flex_attention` 拿着物理页号
去物理 KV 张量取数据，结果才是正确的。

### 一句话总结

> `create_block_mask` 输出的是**连续地址空间中的块序号（逻辑块号）**；
> `flex_attention` 需要的是**物理 KV 张量上的实际块偏移（物理页号）**。
> PagedAttention 的核心就是两者不一致，所以必须用页表做一次显式转换。

---

## `page_size` 与 `BLOCK_SIZE`：两个独立的粒度

`page_size` 属于 **KV Cache 内存管理层**（每页存多少 token），`BLOCK_SIZE`
属于 **FlexAttention 计算层**（每次计算处理多少 token）。两者概念上完全解耦——
`mask_mod` 在单 token 粒度工作，无论 `page_size` 和 `BLOCK_SIZE` 取何值，
逻辑翻译 `physical_kv_idx → logic_kv_idx` 始终正确。

然而，上述 `gather` 技巧隐含了一个对齐约束：

> `flex_attention` 内部用 `kv_indices * BLOCK_SIZE` 定位物理 KV 张量的起始偏移；
> `page_table` 的索引单位是 `page_size`。  
> **只有当 `page_size == BLOCK_SIZE` 时，"物理页号 × BLOCK_SIZE" 才恰好等于
> 该页在 KV 张量上的字节偏移，`gather` 替换才是正确的。**

### 为什么示例在 CPU 上不报错

示例的 `KV_LEN = 12`（极短），`create_block_mask` 在 CPU 上会自动将 `BLOCK_SIZE`
退化为 1。此时每个 token 独占一个"块"，`kv_indices` 直接是 token 级的物理地址，
`page_size=2` 与 `BLOCK_SIZE=1` 的不一致被 `mask_mod` 的 token 级过滤掩盖，
结果碰巧正确。若在 GPU 上以 `BLOCK_SIZE=128` 运行且 `page_size ≠ 128`，
上述 `gather` 替换将产生错误的物理地址。

### 生产级做法（二选一）

| 方案 | 做法 | 适用场景 |
|------|------|----------|
| ① 对齐 | 令 `page_size == BLOCK_SIZE`，按 FlexAttention 的 block size 对齐分页（vLLM 的实际做法） | 需要充分利用 `block_mask` 的整块跳过优化 |
| ② 纯 mask_mod | 不替换 `kv_indices`，完全依靠 `mask_mod` 做 token 级过滤 | `page_size` 可任意选择；代价是更多 partial block，跳过优化效果下降 |

---

## attention-gym 的生产级实现

[attention-gym](https://github.com/meta-pytorch/attention-gym) 的
`attn_gym/paged_attention/paged_attention.py` 提供了一个完整的 `PagedAttention`
类，其 `convert_logical_block_mask()` 方法正是上述 gather 思路的生产级版本。
与本文极简示例相比，它有以下几处关键差异：

### 1. 显式断言 `page_size == BLOCK_SIZE`

```python
if block_mask.BLOCK_SIZE[1] != self.page_size:
    raise RuntimeError(
        f"Expect block_mask has the same column block size as page_size"
        f"but got size={block_mask.BLOCK_SIZE[1]} and size={self.page_size}"
    )
```

attention-gym **主动校验**这一约束，违反时立即报错，而不是让错误的物理地址悄悄
滑入计算。

### 2. `new_kv_indices` 列数扩展到 `n_pages`

```python
new_kv_indices = torch.zeros((B, H, ROWS, self.n_pages), dtype=torch.int32, device=device)
new_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
    torch.gather(page_table, 1, block_mask.kv_indices.view(B, -1).to(torch.int64))
    .view(block_mask.kv_indices.shape)
    .to(torch.int32)
)
```

逻辑块号的最大值是 `MAX_BLOCKS_IN_COL - 1`，物理页号的最大值则可达
`n_pages - 1`（远大于前者）。若直接在原始形状上替换，`_ordered_to_dense`
等内部函数访问索引时会越界。attention-gym 将列维度预先扩展到 `n_pages`，
再把 gather 结果填入前 `MAX_BLOCKS_IN_COL` 列，其余保持 0（永不被访问）。

本文示例在 CPU 上因 `BLOCK_SIZE=1`、`n_pages` 极小而未触发越界，生产环境必须
按此处理。

### 3. 同步替换 `full_kv_indices`

`BlockMask` 除 `kv_indices`（稀疏块）外还有 `full_kv_indices`（整块参与
attention 的块）。attention-gym 用同一 gather 逻辑对两者同时做物理化：

```python
if block_mask.full_kv_num_blocks is not None:
    new_full_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
        torch.gather(page_table, 1, block_mask.full_kv_indices.view(B, -1).to(torch.int64))
        .view(block_mask.full_kv_indices.shape)
        .to(torch.int32)
    )
```

若只替换 `kv_indices` 而忽略 `full_kv_indices`，全块路径仍会走到错误物理地址。

### 4. 用 `BlockMask.from_kv_blocks()` 构造新对象

attention-gym 不依赖内部 `_replace` API，而是用公开的工厂方法：

```python
return BlockMask.from_kv_blocks(
    new_kv_num_blocks,
    new_kv_indices,
    new_full_kv_num_blocks,
    new_full_kv_indices,
    block_mask.BLOCK_SIZE,
    new_mask_mod,
    seq_lengths=(block_mask.seq_lengths[0], self.n_pages * self.page_size),
)
```

同时将 `seq_lengths` 的 KV 维度从逻辑空间长度更新为物理空间总长
`n_pages * page_size`，确保 flex_attention 的边界检查与物理张量尺寸一致。

### 5. 小结：本文示例与 attention-gym 的对比

| 维度 | 本文示例 | attention-gym |
|------|----------|---------------|
| `page_size == BLOCK_SIZE` 约束 | 隐含，CPU 下因退化而掩盖 | 显式 RuntimeError |
| `kv_indices` 列数 | 保持原形状（小序列下侥幸不越界） | 扩展到 `n_pages` |
| `full_kv_indices` | 未处理 | 同步 gather |
| 构造新 BlockMask | `block_mask._replace`（内部 API） | `BlockMask.from_kv_blocks`（公开 API） |
| `seq_lengths` KV 维度 | 未更新 | 更新为物理空间总长 |

---

## 适用场景与局限

**适用**：
- 在现有 FlexAttention 环境下快速验证 PagedAttention 语义正确性。
- 作为自定义 KV Cache 管理策略（如前缀共享、多租户复用）的实验起点。

**局限**：
- `mask_mod` 闭包在每个 token 粒度触发，对极大 batch 或极宽 beam 有性能开销。
- 生产级实现（如 vLLM）在 C++/CUDA 层做了更细粒度的内存管理，本方案为纯
  Python 层等价实现，不能直接替代生产调度器。
- `block_mask._replace` 是内部 API，跨版本需确认字段名。
- `block_mask.kv_indices` 的 `gather` 替换隐含 `page_size == BLOCK_SIZE` 约束。
  在 CPU 小序列示例中，`BLOCK_SIZE` 自动退化为 1 掩盖了该约束；GPU 生产环境下
  若两者不一致会产生错误结果（参见上方"`page_size` 与 `BLOCK_SIZE`"节）。

---

## 参考

- PyTorch FlexAttention 文档：<https://pytorch.org/docs/stable/nn.attention.flex_attention.html>  
- vLLM PagedAttention 论文（Kwon et al., 2023）：<https://arxiv.org/abs/2309.06180>
