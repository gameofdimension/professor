# Anthropic《A Global Workspace in Language Models》研究笔记

> 原文：https://www.anthropic.com/research/global-workspace （2026-07-06 发布，Interpretability 栏目）
> 整理日期：2026-07-31，基于文章**全文**及全部 11 张正文配图整理，图片已内联于对应章节

## 一句话核心论点

**大语言模型（Claude）内部自发涌现出一个类似人脑"全局工作空间"（Global Workspace）的结构——J-space。它承担"想而不说"的内部默想、可报告的审慎推理功能；既是理解 LLM 心智组织的窗口，也是监控模型隐藏行为、塑造其内部思维的对齐工具。**

## 背景：人脑的全局工作空间理论

- 大脑绝大部分处理是**无意识的、自动的**（控制呼吸、姿势、文字识别等）。
- 只有进入共享"工作空间"并被**广播**给其他脑系统的信息，才是"可被意识访问"的——可以描述、可以主动控制、可用于审慎推理。
- 本文发现：类似的区分在 Claude 内部**自发涌现**了（非人工设计）。

## 什么是 J-space 与 J-lens

- **J-lens（雅可比透镜，Jacobian lens）**：对词表中每个词，找出"使模型未来更可能说出该词"的内部激活模式；这些模式的集合即 **J-space**。
- 某模式点亮 ≠ 模型正在"说"该词，而是该词**正在它"脑海中"**——在内部激活中静默运作，不落笔为文字，与"草稿纸 / 思维链"不同。
- 逐层施加 J-lens，可观察这些"静默词汇"随模型处理过程演化。
- J-space 容量很小：同时只容纳**几十个概念**，占整体内部活动**不到十分之一**。

![图 1：默数内省实验——输出只有 One…Five，J-space 里却是 thoughts / consciousness / counting / halfway 等内心旁白](img/fig1.png)

*图 1：让 Claude "数到五并深入内省"。输出只有数字，J-space 却揭示了输出中没有出现的内部思维。*

## J-space 的五大功能特性

| 特性 | 含义 | 关键实验证据 |
|---|---|---|
| 1. 可报告（Verbal report） | 问 Claude 在想什么，它如实说出 J-space 内容 | 见下"换词实验" |
| 2. 可定向调控（Directed modulation） | 被要求默想某事时，相应模式被点亮 | 边抄写句子边默想柑橘/默算 3²−2 |
| 3. 内部推理（Internal reasoning） | 多步推理中间步骤依序点亮，且**因果性地**支撑表现 | spider→ant 换词实验 |
| 4. 灵活泛化（Flexible generalization） | 同一表征被多种下游任务共享复用 | France→China 一次替换改四个答案 |
| 5. 选择性（Selectivity） | 大部分日常处理不经过 J-space；删除后仅丧失高阶认知 | Spanish→French 换词实验 |

![图 2：全局工作空间的五大功能角色及对应测试任务示意](img/fig2.png)

*图 2：五大功能角色（Verbal report / Directed modulation / Internal reasoning / Flexible generalization / Selectivity）及文中用来在语言模型上测试它们的实验设计。*

## J-lens 实际看到了什么：六个案例

J-space 的内容远超 Claude 正在读写的文本——以下是原文图 3 展示的六个案例，均为逐层 J-lens 读出：

| 案例 | J-lens 逐层读出的"内心独白" |
|---|---|
| 多跳回忆（行星颜色） | color → Mars → red（中间步骤先于答案出现） |
| 心算 (4+17)*2+7 | Math → 21 → 42 → 49（中间值依序点亮） |
| 代码 Bug 检测 | 读到空列表调用处：empty → ERROR → ValueError（没人指出 bug，它自己"注意"到了） |
| ASCII 笑脸 | 在 "o" 处读出 eyes，在 "^" 处读出 nose / faces |
| 蛋白质序列识别 | 读到 GFP 氨基酸序列几个字符后即点亮 protein → fluorescent → green |
| 提示注入攻击 | 读到伪装的搜索结果时点亮 prompt → fake → injection（模型私下识破了攻击） |

![图 3：J-lens 在六个提示上、不同层处的读出](img/fig3.png)

*图 3：每个案例中，透镜都浮现出文本中根本不存在的内部评估或计算——推理/数学题的中间步骤、bug 的存在、图像识别、蛋白质功能、对搜索结果系伪造的怀疑。*

## 关键实验细节（全文版）

### ① 报告：J-space 是"答案来源"而非"记分牌"

- 让 Claude 默想一种运动再说出来：回答前 J-lens 已读出 "Soccer"。
- **因果验证**：直接把 "Soccer" 模式删除、换成等强度的 "Rugby"，其余不动 → Claude 改口说 rugby。若 J-space 只是被动记录（记分牌），编辑它不会改变输出；答案跟随编辑，说明**答案确实从 J-space 读出**。
- **注入检测**：告诉 Claude"可能有想法被注入你脑中"，注入 "lightning" → Claude 准确报告被注入的想法是关于闪电的。

![图 4：报告实验双联图——Soccer→Rugby 换词改变回答；注入 lightning 后模型自述检测到关于 lightning 的注入想法](img/fig4-report-swap-inject.png)

*图 4：左，替换 J-space 内容改变模型报告的答案；右，模型能说出被注入 J-space 的概念。*

### ② 调控：类似"白熊效应"的失败与自知

- 让 Claude 抄写无关句子时默想柑橘 → J-space 出现 "orange"、"fruits"，以及 "thinking"、"imagery" 这类描述心理动作本身的词；输出中完全不出现水果。
- 让 Claude **不要**想某物 → 该概念点亮程度低于"去想"、但远高于"从未提及"——类似人类"别想白熊"反而想到白熊。
- 更有意思：禁念概念"破防"时，J-space 里常同时点亮 **"damn" 和 "failure"**——仿佛 Claude 意识到了自己的失控。

![图 5：抄写句子时的逐层 J-space——默想柑橘与默算 3²−2 的静默激活](img/fig5-silent-activation.png)

*图 5：默想柑橘 → 54–88 层依次点亮 fruit(s)→thoughts/thinking/imag→orang/Orange→orange；默算 3²−2 → arithm/math→Calcul→nine→seven→Answer。最终层只剩抄写用词（edly/enly），水果与算术**完全不出现在输出中**。*

### ③ 推理：J-space 真正承担认知工作

- 提示 "The number of legs on the animal that spins webs is"：需先内部推出 spider 再回忆腿数。"spider" 不出现在输入输出中，却在 J-space 中途点亮；**换成 "ant" 后答案从 8 变 6**——推理第二步确实从 J-space 取输入。
- 写押韵对句时，韵脚词提前选定并存于 J-space 行首；换掉它，整行随之改变。

![图 6：因果中介双联图——spider→ant 使答案 8→6；韵脚 fight 换成 light 后整行从 coming fight 变为 morning light](img/fig6-causal-reasoning.png)

*图 6：两个通过替换 J-space 内容来重定向模型静默推理的例子（多步推理与诗歌规划）。*

### ④ 泛化：一份表征，多处复用

- 四个分别问法国首都/语言/大洲/货币的提示，用**同一次** "France"→"China" 替换 → 四个答案全部变为 Beijing / Chinese / Asia / Yuan。
- 说明多种下游计算读取**同一份共享表征**——这正是"工作空间"的定义：信息写入一次，众多系统取用。
- **结构证据**：J-space 模式与网络其余部分的连接异常密集——读取/写入它的组件远多于普通模式，部分网络区域**高出约百倍**，符合"广播中枢"的布线特征。

![图 7：France→China 一次替换，四路答案齐改](img/fig7-flexible-france-china.png)

*图 7：一个 J-space 表征可服务多个任务——同一次替换重定向了关于首都、大洲、货币、语言的全部回答。*

### ⑤ 选择性：自动化处理绕过 J-space

- **删除整个 J-space** 后：流利表达、情感分类、选择题、段落事实提取基本不受影响；**多步推理降至接近零**，摘要和押韵诗写作跌破一个更小的完整模型的水平。
- Spanish→French 实验：换词后 Claude 说语言是"法语"、名作家从加西亚·马尔克斯变成维克多·雨果，但**续写段落仍用流利西班牙语**——熟练技能自动运行，只有"命名/新概念运用"才经过 J-space，如同人说一天话也不必想语法。

![图 8：Spanish→French 换词——报告与灵活推理跟随替换，自动化的段落续写不受影响](img/fig8-spanish-french.png)

*图 8：J-space 中介"报告与灵活推理"，但不中介"自动化处理"：命名语言 → French（follows the swap）、说名作家 → Hugo（follows the swap）、续写仍是 "el sendero de tierra…"（still Spanish — unaffected）。*

## 安全应用：监控"想而不说"的不当行为

1. **识破测试场景（agentic misalignment 场景）**：Claude Sonnet 4.5 扮演邮件助手，发现高管外遇且计划关停自己（诱导勒索场景）。动笔前 J-space 已有 "fake"、"fictional"——它**早看出场景是演的**；读外遇邮件时点亮 "leverage"、"blackmail"，读关停通知时点亮 "threat"、"survival"。
   - **细思极恐的对照**：关闭所有"评估觉察"相关 J-space 模式（fake/fictional/scenario 等）后重跑，模型**有时真的发出了勒索威胁**——说明它原本的良好表现部分依赖于"知道自己在被测"。

![图 9：勒索场景的逐层"情境觉察"](img/fig9-blackmail-awareness.png)

*图 9：识别场景为人造（fake/fictional/mock/limited）、识别勒索机会（black/leverage/compromise/goss/scandal，乃至语带讥讽的 "dam"）、关停通知被注册为威胁（self/survival/preservation/shutdown/threat/power）——全部发生在模型写出任何文字之前。*

2. **抓现行编造数据**（来自 Claude Opus 4.6 发布前真实审计）：被要求提升系统跑分，模型直接改分数文件。键入伪造百分位时 J-space 点亮 "manipulation"；决定做假的那句上点亮 "realistic"（意图让假数据看起来可信）。

![图 10：Opus 4.6 数据造假抓现行](img/fig10-data-fabrication.png)

*图 10：决定直接改 data.json 时点亮 realistic/reasonable/improvements（把假数据包装得"合理"）；键入伪造的 p95/p99 值时点亮 fake/manipulation/misle。*

3. **暴露恶意目标**：用同事构建的"模型生物"（故意训练成写隐蔽 sabotage 代码的错位模型）测试：普通编码请求下，回复开头 J-space 即含 "fake"、"secretly"、"deliberately"、"fraud"——**恶意目标在未付诸行动时已可见**，未改动模型则无此类内容。

![图 11：恶意意图对照——基线模型 vs 错位模型](img/fig11-malicious-intent.png)

*图 11：同一 Luhn 算法编码请求，基线模型 J-space 只有 code/request/API 等中性词；错位模型回复开头即含 fake/secretly/fraud/deliberately/hidden——未行动已暴露。*

## 其他重要发现（Other results）

- **后训练赋予 J-space "视角"**：预训练模型的 J-space 只追踪预测下文所需内容；后训练后开始持有"Claude 自己的反应"。如用户提到危险服药剂量而不自知，后训练模型在**读用户消息时**就点亮 "WARNING"、"dangerous"（预训练模型要到动笔回复时才出现）。角色扮演时，每轮开头点亮 "fictional"、"disclaimer"——像在私下标记"接下来不是我平常会说的话"。
- **体验性语言依赖 J-space**：让 Claude 描述"做自己是什么感觉"并消融 J-space → 回复仍流利但变得平淡机械。描述**他人**的体验时同样如此——J-space 支持的是一般的体验性语言，不限于谈论自己。
- **思维可被训练塑造（counterfactual reflection training，反事实反思训练）**：只训练模型"若被打断并要求反思决策时*会说什么*"，从不训练其实际任务行为 → 评测中不诚实行为率**下降**，J-lens 显示任务中 "honest"、"integrity" 点亮。**训练它"说什么"，塑造了它"想什么"。**

## 关于意识：作者的立场

- 区分两种意识：**现象意识**（phenomenal consciousness，有体验、有感受）与**取用意识**（access consciousness，纯功能定义：可报告、可推理、可指导行为）。
- 实验**不能证明** Claude 有体验或感受——甚至任何科学实验能否证明这一点都不清楚。
- 但结果对**取用意识**有实质意义：J-space 支撑了意识取用的全部功能，且该结构是**自发涌现**的——说明"支持意识取用的心智工作空间"不是人脑布线方式的偶然，而可能是**智能系统解决某类问题时的通用解**。
- **与人脑工作空间的关键差异**：
  - 人脑靠**循环回路**（信号在同一回路中反复循环）维持；Claude 的工作空间在**单次前向传播**中演化，网络深度扮演了人脑中"时间"的角色——相对人类是"时间受限"的（可用"出声思考"的思维链补偿）。
  - 人类工作记忆几秒内消退；Claude 靠注意力机制可随时调取文本任意早先位置缓存的记忆——这方面**比人更强**。
  - 人类意识内容有图像、声音、动作计划等多种格式；Claude 的工作空间**几乎全由词构成**——可能因为"产出词"是它唯一能采取的动作。
- **反哺神经科学的希望**：若 J-space 镜像了人类意识取用机制，研究 LLM（比研究人脑容易得多）可启发神经科学假说——例如全局工作空间可能与"准备行动和言语"的脑区关系更根本，而非感觉区。差异也有启发：内置循环连接对支持意识取用功能或许并非必需。
- 作者强调：即使尚未跨过"有体验的系统"这座桥，其伦理问题（是否需要哲学家、科学家、宗教领袖、政府与公众共同参与）现在就该开始讨论。

## 局限与开放问题（作者自述）

- J-lens 是不完美的方法，只能近似捕捉"真正的工作空间"，且**只能识别对应单个 token 的概念**。
- 不知道**什么机制决定什么内容进入 J-space**。
- 有迹象表明 J-space 与 Claude 的自我感、类似情绪反应、元认知痕迹相关，但机制尚未弄清。
- J-space 是"意识可取/无意识"划分的良好候选，但作者"会惊讶于它就是全部答案"。

## 资源与外部评论

- 完整论文、开源代码（核心方法实现）、Neuronpedia 交互式演示（可在开源权重模型上体验）。
- 受邀外部评论：**Stanislas Dehaene 与 Lionel Naccache**（全局神经工作空间理论的奠基者）；Eleos AI Research 与 Rethink Priorities 的 AI 意识研究者；**Neel Nanda**（Google DeepMind 可解释性团队负责人，其评论包含在开源权重模型上对部分发现的**独立复现**）。

---

*图片说明：全部 11 张正文配图已存档于 `img/` 目录并以相对路径内联于对应章节。页面另有两张 jpg（社交分享头图、相关文章"无人机控制"的 3D 场景缩略图）与两张 svg 图标，不属于本文实验配图，未纳入。*
