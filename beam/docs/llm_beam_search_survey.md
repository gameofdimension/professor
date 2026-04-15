# vLLM、TensorRT-LLM 及主流 LLM 推理引擎 Beam Search 调研

> 更新时间：2026-04-15  
> 说明：不同版本能力差异较大，落地前请以目标版本官方文档/Release Note 为准。

## 结论速览

- **vLLM**：支持 Beam Search，适合“在线服务 + 较高吞吐”场景。
- **TensorRT-LLM (TRT-LLM)**：支持 Beam Search，且在 NVIDIA GPU 上通常有更强推理性能上限。
- **Transformers (HF)**：Beam Search 功能最完整，常作为算法验证基线。
- **TGI / SGLang / Ollama / llama.cpp(Server 常规路径)**：更偏向采样与并发吞吐，Beam Search 通常不是主推路径（部分场景可用或需额外改造）。

## 对比表（Beam Search 相关）

| 引擎 | Beam Search 支持情况 | 常见参数/入口 | 特点与注意点 |
|---|---|---|---|
| vLLM | 支持 | `SamplingParams(use_beam_search=True, best_of=beam_width, length_penalty, early_stopping)` | 与其连续批处理结合后，在线吞吐表现通常好于纯离线脚本；Beam 会显著增加 KV Cache 与调度开销。 |
| TensorRT-LLM | 支持 | 运行时常见 `beam_width`（以及长度惩罚、早停等配置） | 对 NVIDIA 平台优化深，适合低延迟/高吞吐生产部署；部署链路复杂度高于纯 Python 方案。 |
| Hugging Face Transformers | 强支持（最全） | `generate(num_beams, num_return_sequences, length_penalty, early_stopping, no_repeat_ngram_size...)` | 功能覆盖最完整，便于算法实验；大规模服务吞吐通常不及专用引擎。 |
| Hugging Face TGI | 非主推（以采样为主） | API 主要聚焦采样参数 | 优势在多租户服务化与易运维；若强依赖 Beam Search，常改用 vLLM/Transformers/TRT-LLM。 |
| SGLang | 非主推（以高吞吐采样为主） | 常用接口偏采样/并发执行 | 适合高并发生成任务；Beam 需求需要先做版本级能力确认。 |
| Ollama | 非主推 | 常见 `options` 以采样参数为主（如 temperature、top_p），Beam 相关参数通常无统一入口 | 本地部署简单，但 Beam Search 可控性和参数覆盖不如专业推理引擎。 |

## vLLM 重点

1. **开启方式**  
   一般通过 `use_beam_search=True`，并将 `best_of` 作为 beam 宽度（具体字段以版本文档为准）。
2. **推荐实践**  
   - Beam 模式下优先使用较低温度（常见为 0）避免与采样语义混杂。  
   - 先做 `beam_width`（如 2/4/8）压测，观察时延、显存和质量曲线。
3. **常见限制**  
   Beam Search 往往会显著提高显存与解码步延迟，吞吐压力场景需谨慎放宽 beam 宽度。

## TensorRT-LLM 重点

1. **开启方式**  
   运行时配置 `beam_width`，并结合长度惩罚/早停策略做质量调优。
2. **推荐实践**  
   - 与模型量化、分页 KV Cache、并行策略联合压测。  
   - 将 Beam 宽度作为 SLA 维度（延迟、吞吐、成本）的一部分统一评估。
3. **常见限制**  
   工程化门槛较高（构建引擎、版本匹配、部署链路复杂），但上限通常更高。

## 选型建议（按场景）

- **算法评估/离线实验优先**：Transformers（最灵活）。  
- **在线服务且要 Beam Search + 易集成**：vLLM。  
- **NVIDIA 生产环境追求极致性能**：TensorRT-LLM。  
- **主要追求采样吞吐与服务化**：TGI/SGLang（Beam 需求需谨慎评估）。

## 建议的最小验证清单

1. 固定一组提示词与评测集（准确率/ROUGE/业务指标）。
2. 对 `beam_width = 1,2,4,8` 做 A/B 压测。
3. 记录：P50/P95 延迟、tokens/s、显存峰值、输出质量。
4. 对比同等硬件下 vLLM 与 TRT-LLM 的成本-效果曲线。

## 参考入口（建议复核）

- vLLM 文档（SamplingParams / OpenAI compatible server）：  
  https://docs.vllm.ai/
- TensorRT-LLM 文档（runtime generation config / beam width）：  
  https://nvidia.github.io/TensorRT-LLM/
- Hugging Face Transformers `generate` 文档：  
  https://huggingface.co/docs/transformers/main_classes/text_generation
- Hugging Face TGI 文档（生成参数能力边界）：  
  https://huggingface.co/docs/text-generation-inference/
- SGLang 文档：  
  https://docs.sglang.ai/
- Ollama 文档：  
  https://github.com/ollama/ollama/tree/main/docs
