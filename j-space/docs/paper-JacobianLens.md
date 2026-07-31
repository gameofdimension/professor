# 论文精读：The Jacobian Lens 原理与开源复现方法

> 论文：*Verbalizable Representations Form a Global Workspace in Language Models*
> 链接：https://transformer-circuits.pub/2026/workspace/index.html
> 代码：https://github.com/anthropics/jacobian-lens ｜ 演示：https://www.neuronpedia.org/jlens
> 整理日期：2026-07-31，基于 WebBridge 读取的论文全文（§2 Methods、§A.2、§A.5–A.7）及官方仓库 README

---

## 一、J-lens 的原理（§2.1）

### 1. 核心思想：用"一阶因果效应"刻画激活

对第 ℓ 层残差流 $h_\ell$ 在 token 位置 t 施加一个小扰动，它会穿过剩余各层，影响最终层**所有后续位置** t′ ≥ t 的残差流 $h_{\text{final},t'}$。一阶近似下这是线性关系，由雅可比矩阵描述：

$$\frac{\partial h_{\text{final},t'}}{\partial h_{\ell,t}}$$

再复合 unembedding 层，就得到"该扰动对位置 t′ 输出 logits 的一阶影响"。直觉上：**J-lens 回答的不是"这个激活是什么"，而是"这个激活平均而言倾向于让模型说出什么词"**。

### 2. 关键一步：跨上下文平均

单个 prompt 上算出的雅可比混合了两种东西：

- 模型**言说某概念的通用倾向**（想要的）
- 该概念在**当前语境的具体用法**（噪声）

为分离前者，对提示语料、源位置、目标位置三者求期望，每层得到一个 $d_{\text{model}} \times d_{\text{model}}$ 矩阵：

$$J_\ell \;=\; \mathbb{E}_{\,t,\; t' \geq t,\; \text{prompt}} \left[ \frac{\partial h_{\text{final},t'}}{\partial h_{\ell,t}} \right]$$

- 语料：**1000 条**预训练分布风格的 prompt（每条 128 token）
- 期望范围：源位置 t、同上下文中所有后续位置 t′、整个语料

### 3. 读出方式

$$\text{lens}(h_\ell) \;=\; \text{softmax}\big(W_U \,\cdot\, \text{norm}(J_\ell \cdot h_\ell)\big)$$

相当于**用一个线性矩阵 $J_\ell$ 替换掉第 ℓ 层之后的所有层**，再走过模型自己的归一化 + unembedding，得到词表上每个 token 的分数，排序取 top 即为该激活的可读描述。

- $W_U J_\ell$ 的每一行 = 一个 **J-lens 向量**：词表单个 token 对应的残差流方向。

---

## 二、J-space 的定义（§2.3）

- 每层的 J-lens 向量构成**超完备集**（n_vocab 个向量 > d_model 维），分解不唯一。
- 经验上同时只有少量 J-lens 向量强激活，因此 **J-space 定义为：能表示为不超过 k 个 J-lens 向量的稀疏非负组合的所有点的集合**（几何上 = 若干 k 维锥的并）。
- k 有点任意，论文通常取 **k ≤ 25**（经验观察到的同时有意义的激活数，见 §4.2）。
- 操作上用**梯度追踪（gradient pursuit）**做稀疏非负分解，把一个激活近似为 k 个 J-lens 向量的组合，系数即"局部 J-space 坐标"。
- J-space 分量通常只占激活总方差的**不到 10%**。
- 叠加假说（superposition）视角：J-lens 向量是模型特征方向"稀疏框架"中的一个**按 token 索引的子框架**。

---

## 三、读与写：J-lens 的用法（§2.5）

### 读（Reading）

1. **全词表排序读出**：$\text{lens}(h_\ell)$ 排序取 top token（论文中所有 "top lens tokens" 均指此）。
2. **单概念探针**：直接算 $h_\ell$ 与某个 J-lens 向量 $v_t$ 的内积/余弦相似度，判断特定概念是否超过阈值。
3. **稀疏分解盘点**：梯度追踪解出 k 个向量的稀疏非负组合，得到去冗余的"活跃概念清单"（用于 §4.2 的占用率与方差分析）。

### 写（Writing）

1. **沿 J-lens 向量 steering**：$h \leftarrow h + \alpha\, v_t$；α 为负或投影掉该分量即**消融**（用于压制某概念、或整体压制 top-k J-space 内容）；正向 steering 用于"注入概念"的内省检测实验。
2. **透镜坐标修补（patching in lens coordinates）**：源 token s → 目标 token t 的概念交换。构造 $V = [v_s\; v_t]$，读出坐标 $c = V^\dagger h$（伪逆），令 $h_{\text{patched}} = h + V(\sigma(c) - c)$，其中 σ 交换两个坐标（可选缩放 α）。与 $\text{span}\{v_s, v_t\}$ 正交的分量不变——这就是 Soccer→Rugby、spider→ant、France→China 等换词实验的数学操作。

### 写操作的展开解释与直觉

**数学上在做什么。** steering 中 $v_t$ 是 $W_U J_\ell$ 的第 t 行。按定义，扰动 $\delta h$ 对 token t 的 logit 的一阶影响为 $\Delta \text{logit}_t = v_t \cdot \delta h$——要让该 logit 涨得最快，沿 $v_t$ 走就是梯度上升方向（每单位长度收益最大）。因此 steering 不是"随便找个方向推一推"，而是**对"说出 t"做了一步最优的一阶推动**；α 为负或投影掉分量即反向操作（消融）。换词操作三步各有明确含义：$c = V^\dagger h$ 用伪逆回答"只用 v_s、v_t 两个方向怎样组合最逼近 h 在该平面内的部分"（因为 J-lens 向量超完备、非正交，不能直接拿内积当坐标）；$\sigma(c)$ 保持"概念总量"不变、只改归属（soccer 的分量数值原封转给 rugby）；只加 $V(\sigma(c)-c)$ 使 h 中与 $\text{span}\{v_s,v_t\}$ 正交的部分分毫不动——外科式最小干预。

**为什么符合直觉。**

1. **读写共用一套坐标轴，如同对同一群神经元既记录又刺激。** 神经科学验证"这群神经元编码 X"的黄金标准是双管齐下：记录时 X 出现它放电（读），刺激它则个体表现出 X（写）。J-lens 向量天然满足此对偶性——内积 $v_t \cdot h$ 是读（t 在脑海中有多强），加 $\alpha v_t$ 是写（把 t 放进脑海）。同一批方向既充当感知通道又充当控制旋钮，这正是"工作空间"的题中之义：信息从该通道广播出去，也只能通过该通道被改写。许多 steering 方法"读用一套探针、写用另一套向量"，两者对不上，解释力就弱。
2. **坐标轴自带语义标签，像每个推子都标了字的调音台。** 普通激活空间有几千个无名维度；J-lens 坐标系里每个轴就是一个词（"法国"轴、"ERROR"轴），且功能由构造保证（它就是"让模型说这个词"的方向），不是事后人工标注。干预从"高维空间瞎转旋钮"变成"把标着 France 的推子拉下、把标着 China 的推上"。
3. **换词 = 只改广播内容，不改广播线路。** 若下游各系统各自存副本，改工作空间一处最多影响一个任务；结果 France→China 一次替换四个答案齐改，证明大家确实在读同一块白板。消融 "spider" 使答案 8→6，证明白板内容因果性地支撑推理而非记分牌——若是记分牌，擦掉它比赛照样进行。
4. **正交互补不动 = 最小惊讶原则。** 好的干预只动想动的东西；正交保护保证"除了把 soccer 换成 rugby，其他什么都没碰"，让结果归因干净。

**直觉的边界（论文自述）。** 一切都是一阶线性近似 + 跨上下文平均：$v_t$ 是"平均而言让模型说 t"的方向，在具体语境中未必最优；只能操作单 token 概念（多词概念见 §A.9 扩展）；向量非正交使伪逆坐标是近似，故换词有时需加缩放因子 α 调强度。

一句话：**J-lens 把"影响输出"和"概念标签"统一在同一组方向上——读是投影，写是平移，换词是交换坐标，消融是清零分量，全部操作都在一本自带词表的词典里进行，且被一阶因果保证所约束。**

### 报告口径

- 全文结果在残差流 **25 个等间距层**上报告，重新索引到 [0–100] 区间，层号即百分比。
- 默认模型 Claude Sonnet 4.5；关键结果在 Haiku 4.5 / Opus 4.5 上 corroborate，部分分析用 Opus 4.6。

---

## 四、方法学细节与消融（§A.7）——复现时最重要的工程选择

J-lens 有三个独立的设计轴：

### 1. 对什么求梯度

- **目标层**：默认取**倒数第二层**残差流（反向传播省略最后一个 transformer block）——因为末块高度专门化于校准 next-token 预测、语义含量低，纳入会增加噪声伪影。
- **注意力梯度**：默认标准反传（扰动可改变下游注意力模式）；另有 **frozen-QK 变体**（Q/K 投影梯度置零，只保留 value 通路），分离"搬运什么"与"关注什么"。加 stop-grad 可**增大因果干预效果**。
- **目标位置**：默认平均所有 t′ ≥ t（混合当前+未来）；另有 self-only（t′=t，接近 logit lens 精神）与 future-only（t′>t，隔离"广播"分量）。

### 2. 聚合方式

- prompt 内：对位置取均值，可剔除范数离群位置（>Nσ）和序列开头若干位置。
- prompt 间：逐元素 mean / median；可按雅可比 Frobenius 范数过滤离群 prompt。
- 消融结论：**各配方相当稳健**；"penultimate + mean 聚合"在提取中间步骤上略优；frozen-QK 提升因果效应。

### 3. 数据

- **数量**：默认 1000 条 × 128 token；扫描显示**仅 10 条 prompt 就已超过 logit lens / tuned lens 基线**，之后缓慢饱和（README 说 ~100 条即可用）。
- **分布**：默认预训练分布；剔除开头 token、剔除非字母数字后继位置等尝试均无显著提升。

### 4. 官方伪代码（论文 §A.7 原文）

```
# Compute J_ℓ for all layers ℓ.
# h_ℓ[t] : residual stream at layer ℓ, position t
# z[t]   : residual stream at the target layer L (by default final)
for each prompt p in corpus:
    run forward pass; cache h_ℓ[t] for all ℓ, t
    # one backward pass per output dimension (in practice batched)
    for i in 1..d_model:
        # inject ∂/∂z_i = 1 at every position, backprop to every layer
        grad_z = e_i ⊗ 1_T          # one-hot in dim i, for every token position
        for each layer ℓ:
            G_ℓ = ∂(Σ_t z[t]) / ∂h_ℓ        # autodiff; shape [T, d_model]
            J_ℓ^(p)[i, :] = mean over positions t of G_ℓ[t, :]
# aggregate across prompts (per element)
for each layer ℓ:
    J_ℓ = mean over prompts p of J_ℓ^(p)
# apply the lens
lens(h_ℓ) = softmax( W_U · norm( J_ℓ · h_ℓ ) )
```

要点：每个 prompt 只需一次前向（缓存所有层残差流）+ d_model 次（实际为批量化的）反向传播；先在 prompt 内对源位置取均值，再跨 prompt 逐元素取均值。

---

## 五、开源复现：官方资源与具体路径

### 1. 论文官方声明（§A.2）

> 为协助复现，我们提供 Jacobian lens 训练与推理的开源实现。仓库中还包括方法学评估（§A.6）及正文全部实验所用的原始 prompt 数据。开源模型上的交互式 J-lens 托管于 Neuronpedia。

- 代码：**https://github.com/anthropics/jacobian-lens**（Apache 2.0；定位是"参考实现"——不维护、不接受贡献、未做性能优化）
- 演示：**https://www.neuronpedia.org/jlens**（开源权重模型上的交互式 J-lens 读出）
- 外部独立复现：Neel Nanda（Google DeepMind）在受邀评论中包含在开源权重模型上对部分发现的独立复现

### 2. 仓库能力（据 README）

- 在**开源权重 decoder transformer** 上拟合（fit）、应用（apply）透镜，并渲染论文同款的 层 × 位置 交互视图。
- **示例用 Qwen**，其他 HuggingFace decoder 可干净适配。
- 不捆绑模型权重与语料（运行时自行下载，遵守各自许可）；`data/` 下的复现与评估 prompt 集是 Anthropic 编写的合成数据，同为 Apache 2.0。

### 3. 复现操作步骤

**安装**

```bash
pip install -e .
```

**应用一个已拟合的透镜**

```python
import transformers, jlens

hf = transformers.AutoModelForCausalLM.from_pretrained("org/model").cuda()
tok = transformers.AutoTokenizer.from_pretrained("org/model")
model = jlens.from_hf(hf, tok)

lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model.pt")
lens_logits, model_logits, _ = lens.apply(
    model, "Fact: The currency used in the country shaped like a boot is",
    positions=[-2])
for layer, logits in sorted(lens_logits.items()):
    print(layer, [tok.decode([t]) for t in logits[0].topk(5).indices])
```

**在自己的模型上拟合透镜**

```python
lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
lens.save("out/jacobian_lens.pt")
```

- 论文规格：1000 条 × 128 token 的类预训练语料；**~100 条即可用**（质量迅速饱和）。
- 耗时大头是模型自身的反向传播（参考实现未优化）；可对 prompt 分片并行跑 `fit()`，再用 `JacobianLens.merge()` 合并。
- 估计器细节（cotangent 先对目标位置求和、再对源位置平均）写在 `jlens/fitting.py` 的 docstring 里。
- 端到端走查见仓库 `walkthrough.ipynb`（加载模型 → 加载/拟合透镜 → 多层应用 → 渲染 slice 页面）。

### 4. 交互视图读法（slice page）

- 每格显示该 (位置, 层) 的 top-1 词，上标为它在全词表的排名。
- 点击格子可选中并"钉住" token，得到排名轨迹曲线与排名热力图。
- 最底行（L = n_layers − 1）是模型实际输出。
- 注意：**约前 1/3 层的读出噪声大、基本不可解读**（§2.2、§4.1）；最后几层会翻转成预测 next-token 的"motor regime"。

---

## 六、一句话总结

**J-lens = 把"中间层激活 → 最终输出"的雅可比在 1000 条语料上取平均，得到一个每层固定的线性矩阵；用它替换后续所有层再过 unembedding，就能读出激活"平均倾向于让模型说什么词"。J-space 则是这些 token 向量的稀疏非负组合所张成的集合。官方已放出 Apache 2.0 参考实现（示例基于 Qwen，适配任意 HF decoder）与 Neuronpedia 交互演示，用约 100 条 prompt 即可在自己的开源模型上拟合出可用的透镜。**

---

## 附录：透镜技术演进对比——Logit Lens → Tuned Lens → J-lens（§2.4、§A.5、§A.6）

### A. 共同的问题

三种技术回答同一个问题：**"中间某层的激活向量 $h_\ell$，用模型自己的 unembedding 解码，对应词表上的哪些词？"** 区别只在于"中间层坐标系 → 最终层坐标系"的运输矩阵 $T_\ell$ 怎么来：

$$\text{readout}(h_\ell) = \text{softmax}\big(W_U \cdot \text{norm}(T_\ell \cdot h_\ell)\big)$$

### B. 三种 $T_\ell$

- **Logit lens**（2020，nostalgebraist）：$T_\ell = I$（恒等），直接用 $W_U$ 乘中间层。残差连接让深层坐标系差异小，所以深层可用；**失效模式是浅层退化**（多跳问题 58 层以下只读出 vah / valea / general 等碎片，荧光蛋白、ASCII 笑脸提示上全程基本是噪声）。零成本，至今是最常用的快速探查工具；在它有效的层里能抓到与 J-lens 大部分相同的工作空间内容。
- **Tuned lens**（2023，Belrose et al.）：$T_\ell$ 是**训练出的逐层仿射映射**，目标为匹配模型最终输出分布——相关性而非因果。**失效模式是"跳步"**：从早期层的微弱信号推断答案类别后，从很早的层起每层级联报最终答案，碾平中间计算（多跳问题从 38 层起一直读 red；诗歌任务读空白符）。附录 A.6 揭示：其早期层的线性部分**实际接近恒等**，对 logit lens 的优势几乎全来自 bias 项——即在"忽略输入、硬猜输出"。
- **Jacobian lens**（2026，本文）：$T_\ell = J_\ell$，真实雅可比的跨上下文平均，**因果**的一阶扰动效应。可看作 logit lens 的"有原则的修正"：$J_\ell$ 恰好是"第 ℓ 层方向 → 最终层方向"的平均线性映射。
- 相关但不同：**Hernandez et al.** 也用雅可比，但推导的是单关系（subject→object，如 "Miles Davis"→"trumpet"）的线性映射；J-lens 把一阶近似用于"激活→模型输出"的整体映射。

### C. 定量对比（§A.6，Sonnet 4.5，六类含已知中间概念的提示分布）

六类分布：多跳（50 题）、多语言（54 题）、运算顺序（55 题）、诗歌（52 题）、拼写错误（96 题）、联想等。

| 指标 | 结果 |
|---|---|
| **中间概念恢复率（pass@k AUC）** | J-lens 在**全部六类**领先；对 logit lens 的优势在多跳/联想上温和，在多语言、运算顺序、诗歌、拼写错误上巨大；tuned lens 垫底（联想、诗歌几乎颗粒无收——这两类读出位置的 next-token 恰好无信息，暴露其"报输出"偏置） |
| **消融因果效应（输出 KL）** | 消融 J-lens 方向引起的输出扰动约为另两者的**两倍**（多跳任务） |
| **换词成功率** | J-lens 坐标交换在**三个模型规模**上都显著更常翻转 top-1 输出 |
| **与真实 next-token 分布的一致度** | tuned lens 每层最优（它为此训练），**J-lens 最差**——作者视之为特性："最能预测输出的方向，并不是最能暴露、驱动计算的方向" |

### D. 按层分区：三种透镜在哪里一致、在哪里分道（§A.6）

| 层区（百分比层号） | 表现 |
|---|---|
| **工作空间前（0–33）** | 三者读出互不一致，全无意义；logit/tuned 向量余弦高达 0.9（tuned 线性部分≈恒等，差异全靠 bias）；J-lens 向量与两者近乎正交（受低秩子空间约束） |
| **早期工作空间（~38–58）** | J-lens 开始恢复中间概念，logit lens 基本不能；J-lens 向量迅速转向（对 logit lens 余弦 20 层内 ~0→~0.7）但读出分歧最大——正是 J-lens 最常"看到别人看不到的东西"的区域 |
| **晚期工作空间（~62–92）** | logit lens 开始读出有用中间步骤，与 J-lens 收敛；tuned lens 仍是局外人 |
| **"运动层"（最后几层）** | 三者完全合流（余弦→1、KL→0、top-1→1）——都在报 next-token 预测 |

### E. 速记表

| | Logit lens | Tuned lens | Jacobian lens |
|---|---|---|---|
| 运输矩阵 $T_\ell$ | $I$ | 训练的仿射映射 | 平均雅可比 $\mathbb{E}[\partial h_{\text{final}}/\partial h_\ell]$ |
| 性质 | 零成本假设 | **相关性**拟合输出 | **因果**一阶扰动效应 |
| 浅层 | 噪声 | 报输出（靠 bias 硬猜） | 前 1/3 层也噪声，工作空间层恢复中间概念 |
| 深层 | 可用 | 可用但跳步 | 可用 |
| 看中间计算 | 深层能看到一部分 | 几乎看不到（被输出挤掉） | **最强** |
| 因果干预（消融/换词） | 弱 | 弱 | **约 2× 因果效应** |
| 需要拟合 | 否 | 需要 | 需要（~100 条 prompt 可用） |
| 一句话 | "直接拿词表去照" | "学会提前报答案" | "测量激活对输出的真实一阶影响" |

> 论文的核心提醒：**"最能预测输出的透镜" ≠ "最能暴露计算的透镜"**。预测行为用 tuned，快速探查深层用 logit，审计内部推理用 Jacobian。
