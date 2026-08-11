# DeepSeek-V4-Flash 目录对比报告

> 对比对象:`/warehouse/DeepSeek-V4-Flash`（无后缀，Preview 预览版）与 `/warehouse/DeepSeek-V4-Flash-0731`（正式版）
> 生成日期:2026-08-11

---

## 一、核心结论

两个目录都是 **DeepSeek-V4-Flash**（284B 总参 / 13B 激活 / MoE）的模型权重,但分属不同版本:

| 维度 | `DeepSeek-V4-Flash` | `DeepSeek-V4-Flash-0731` |
|---|---|---|
| **版本定位** | Preview 预览版 | **官方正式版**（取代预览版） |
| **模型创建时间** (.mv) | 2026-06-21 | 2026-07-31 |
| **下载时间** (mtime) | 2026-08-11 | 2026-08-03 |
| **总大小** | 149 GB | 156 GB |
| **权重分片数** | 46 | 48 |
| **张量数** (index.json) | 69,187 | 72,317（多 ~3,130） |
| **投机解码** | 无 | **带 DSpark 投机解码模块** |
| **reasoning_effort** | 仅 `max`（单档） | `low` / `high` / `max`（三档） |

> 注:目录名无后缀的版本虽是**今天下载的**,但它代表的是**较早的 Preview 版**;带 `-0731` 的才是更新的官方正式版。

**一句话总结**:0731 相对 Preview 的两个主要功能性升级是 —— ① 引入 **DSpark 投机解码模块**（多 ~7GB 权重、2 个分片）,② `reasoning_effort` 扩展为 **三档语义**。模型主体结构（43 层 / hidden 4096 / 256 专家 / 13B 激活 / FP4+FP8 混合）完全一致。

---

## 二、目录结构对比

两者顶层文件清单基本对称,差异在权重分片数和部分配置。

```
共同文件:
  LICENSE, README.md, config.json, configuration.json,
  generation_config.json, tokenizer.json, tokenizer_config.json,
  model.safetensors.index.json,
  encoding/        (2 个不同 + tests/ 相同)
  inference/       (4 个不同 + 3 个相同)
```

| 项目 | Flash | 0731 |
|---|---|---|
| `model-*.safetensors` | 00001 ~ **00046** / 46 | 00001 ~ **00048** / 48 |
| `assets/` 目录 | **有**（`dsv4_performance.png`） | 无 |
| `assets` 引用 | README 引用性能图 | 不引用 |

---

## 三、`config.json` 差异

模型主体配置（`num_hidden_layers=43`, `hidden_size=4096`, `n_routed_experts=256`, `num_experts_per_tok=6`, `head_dim=512`, `quantization_config` FP8 等）**完全相同**。差异:

### 3.1 0731 新增 DSpark 字段
```json
"dspark_block_size": 5,
"dspark_noise_token_id": 128799,
"dspark_target_layer_ids": [40, 41, 42],
"dspark_markov_rank": 256
```

### 3.2 `compress_ratios` 尾部不同
- Flash 结尾:`[..., 4, 0]`（44 项）
- 0731 结尾:`[..., 4, 0, 0, 0]`（尾部多出的 0 对应 DSpark 目标层 40/41/42）

### 3.3 `configuration.json`（框架元信息）
- Flash:`{"framework":"Pytorch","task":"text-generation"}`
- 0731:`{"task":"text-generation"}`（去掉 framework 字段）

### 3.4 相同的配置文件
- `generation_config.json`:**完全一致**（`do_sample=true, temperature=1.0, top_p=1.0`）
- `tokenizer.json` / `tokenizer_config.json`:**完全一致**（6,367,146 B,逐字节相同）

---

## 四、`encoding/` 差异

| 文件 | Flash | 0731 | 状态 |
|---|---|---|---|
| `encoding_dsv4.py` | 27,908 B | 29,001 B | **不同** |
| `README.md` | 8,118 B | 9,168 B | **不同** |
| `test_encoding_dsv4.py` | 3,741 B | 3,741 B | 相同 |
| `tests/*`（8 个 json/txt） | — | — | **全部 md5 一致** |

### 差异内容:`reasoning_effort` 升级为三档

**Flash（旧）** —— 只有 `max` 一档会加 prefix:
```python
REASONING_EFFORT_MAX = "Reasoning Effort: Absolute maximum ..."
assert reasoning_effort in ['max', None, 'high']
if index == 0 and thinking_mode == "thinking" and reasoning_effort == 'max':
    prompt += REASONING_EFFORT_MAX
```

**0731（新）** —— 字典化三档,`low` 为默认且不加 prefix:
```python
REASONING_EFFORT_PROMPTS = {
    "low":  "",                                      # 默认,无 prefix
    "high": "Reasoning Effort: Absolute maximum ...",   # = 旧版 max 文本
    "max":  "Reasoning Effort: Beyond maximum ...",     # 全新,更强
}
DEFAULT_REASONING_EFFORT = "low"
```

**关键变化**:
1. 新增 `low` 档（默认,不加 prefix）,旧版的 `None` 归一为 `low`。
2. `high` 档沿用旧版 `max` 的文本（`Absolute maximum...`）。
3. `max` 档是全新的更强文本（`Beyond maximum — exhaustive, relentless...`）。
4. thinking 模式下任何非 `low` 档位都会加 prefix（旧版仅 `max` 加）。

`README.md` 同步增加了三档 prefix 对照表和 `max` 完整文本。

---

## 五、`inference/` 差异

| 文件 | Flash | 0731 | 状态 |
|---|---|---|---|
| `model.py` | 38,632 B | 45,118 B | **不同**（254 行差异） |
| `generate.py` | 6,296 B | 5,804 B | **不同**（23 行） |
| `convert.py` | 7,075 B | 6,639 B | **不同**（18 行） |
| `config.json` | 991 B | 1,162 B | **不同**（9 行） |
| `README.md` | 951 B | 951 B | 相同 |
| `kernel.py` | 22,198 B | 22,198 B | **相同** |
| `requirements.txt` | 92 B | 92 B | **相同** |

> `inference/` 的全部差异围绕同一条主线:**从普通 MTP（多 token 预测）升级为 DSpark 投机解码**。

### 5.1 `inference/config.json`
0731 新增 5 个字段（其中 `n_mtp_layers` 仅出现在 inference 配置,主 config.json 里没有）:
```json
"n_mtp_layers": 3,
"dspark_block_size": 5,
"dspark_noise_token_id": 128799,
"dspark_target_layer_ids": [40, 41, 42],
"dspark_markov_rank": 256
```
`compress_ratios` 尾部与主 config.json 同样差异（Flash `..., 4, 0]` vs 0731 `..., 4, 0, 0, 0]`）。

### 5.2 `convert.py`（权重名映射表）
- **Flash（旧）** 含一大块 HF 标准命名 → 内部命名的映射（`embed_tokens→embed`, `input_layernorm→attn_norm`, `q_proj→wq`, `kv_a_proj_with_mqa→wkv_a`, `o_proj→wo`, `gate_proj→w1`, `lm_head→head` 等 15 条）。
- **0731** 删除了这批 HF 命名映射（**检查点 key 已与内部命名统一**）,只保留需特殊处理（量化标记 0/1）的几条,并**新增 DSpark 权重**:
  ```python
  "markov_w1": ("markov_w1", 0),
  "markov_w2": ("markov_w2", 0),
  ```

### 5.3 `generate.py`（采样下沉到 model 层）
- **Flash（旧）** 有独立 `sample()`（Gumbel-max）,`generate()` 带 `temperature` 参数自行采样。
- **0731** 删除 `sample()`,改为 `next_token = model.forward(...)[0]`——forward 返回 `(output_ids, logits, main_hidden)` 元组,主模型自己采样。
- 新增 `args.temperature = temperature`、`args.max_seq_len = 64 * 1024`。
- 默认 temperature:Flash `0.6` → 0731 `1.0`。

### 5.4 `model.py`（254 行,DSpark 核心）

**(a) ModelArgs 新增字段**
```python
temperature: float = 1
dspark_block_size: int = 0
dspark_noise_token_id: int = 0
dspark_target_layer_ids: Tuple[int] = tuple()
dspark_markov_rank: int = 256
```

**(b) 索引张量提前转 `int`**:`get_topk_idxs` 等返回 `.int().contiguous()`,减少重复类型转换。

**(c) Block 类注意力改为可插拔**:新增 `attention_cls = Attention` 类属性,`self.attn = self.attention_cls(...)`,forward 透传 `*attn_args`（`main_x`）——使 `DSparkBlock` 能换上 `DSparkAttention`。

**(d) `hc_head` 上移、`head` 简化**:`hc_head`（超连接归一）从多类重复定义统一提到 Block 基类;`head.forward` 签名从 `(x, hc_fn, hc_scale, hc_base, norm)` 简化为 `(x, full_logits=False)`。

**(e) 三个全新类（投机解码核心）**

| 类 | 作用 |
|---|---|
| `DSparkAttention` | 草稿模型注意力,基于滑窗 KV cache + sparse attention（`get_dspark_topk_idxs` 取窗口 token） |
| `DSparkMarkovHead` | Markov 投机头:`markov_w1`（token→低秩 embedding）、`markov_w2`（embedding→vocab logits）,给候选 token 迭代加 bias |
| `DSparkConfidenceHead` | 置信度头:`hidden`+`markov_embed` 拼接投影到 1 维,**proj 用 fp32 存储**保证精度,供投机接受/拒绝 |

**(f) `MTPBlock` → `DSparkBlock` 重写**
- stage 0:有 `main_proj`/`main_norm`,把主模型 target_layer 的 hidden 投影成 `main_x`。
- 最后一个 stage:挂 `markov_head` + `confidence_head`。
- `forward_embed`:用 `noise_token_id` 填充 `block_size` 个草稿输入。
- `forward_head`:逐位置采样 + markov bias,返回 `(output_ids, logits, confidence)`。

**(g) 主模型 forward 返回值改变 + 新增 `forward_spec`**
```python
# Flash:  return logits
# 0731:   return output_ids, logits, main_hidden   # 自采样 + 收集 target layer hidden

def forward_spec(self, input_ids, main_hidden, start_pos=0):
    # 投机解码专用入口:草稿批量生成候选 + 置信度
    ...
    return output_ids, logits, confidence
```

**(h) `sample()` 迁移 + 测试更新**:`sample()` 从 generate.py 搬到 model.py（主模型与 DSpark 共用）,新增 `temperature==0 → argmax` 分支;`__main__` 测试改用 DSpark 配置。

---

## 六、权重文件对比

| 项目 | Flash | 0731 |
|---|---|---|
| 分片数 | 46 | 48 |
| `model.safetensors.index.json` 大小 | 5,371,381 B | 5,602,871 B |
| `weight_map` 张量数 | 69,187 | 72,317 |
| 总大小 | 149 GB | 156 GB |

0731 多出的 ~7GB / ~3,130 个张量 / 2 个分片,对应 **DSpark 草稿模型模块**（`markov_w1`、`markov_w2`、`confidence_head` 的 `proj`、`main_proj` 等）。

---

## 七、完全相同的文件汇总

以下文件在两个目录中逐字节一致:
- `tokenizer.json`（6.37 MB）
- `tokenizer_config.json`
- `generation_config.json`
- `LICENSE`
- `encoding/test_encoding_dsv4.py`
- `encoding/tests/*`（8 个输入/输出测试文件）
- `inference/kernel.py`（22 KB,FP4/FP8 量化算子）
- `inference/requirements.txt`
- `inference/README.md`

**核心 CUDA 算子（`kernel.py`）和依赖未变。**

---

## 八、结论与建议

1. **0731 是 Flash 的升级正式版**,两者模型主体结构完全一致,主体权重应可互通（但需核实分片对应关系）。
2. **主要新增能力**:
   - **DSpark 投机解码**——0731 内置草稿模型（Markov head + Confidence head）,可通过 vLLM 的 `--speculative-config '{"method":"dspark",...}'` 一行启用,加速生成。
   - **`reasoning_effort` 三档**（low/high/max）——精细控制推理深度。
3. **使用建议**:
   - 需要推理加速 / agentic 场景 → 选 **0731**。
   - 只需基础模型、不使用投机解码 → 两者主体等价,但 0731 更新且能力更强,仍建议 0731。
   - 从 Preview 迁移到 0731 时注意:`reasoning_effort` 语义变化（旧 `max` ≈ 新 `high`）,且 inference 代码接口改变（forward 返回元组）。
