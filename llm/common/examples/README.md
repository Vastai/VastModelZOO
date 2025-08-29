# Examples 说明

## Sampling 参数支持说明

|  | **参数** | **说明** | **类型** | **默认值** | **vacc支持情况** |
| --- | --- | --- | --- | --- | --- |
| **生成数量** | **n** | 返回n个不同的回答 | int | 1 | 支持 |
|  | **best\_of** | 返回多个回答，然后选择n个<br>必须大于等于n，默认为n<br>**v1弃用** | int | None | 支持 |
| **生成长度** | **max\_tokens** | 每个输出序列生成的最大token数<br>遇到遇到 `stop`/`EOS` 或 `min_tokens` 未满足，会提前终止 | int | 16 | 支持 |
|  | **min\_tokens** | 在EOS或stop\_token\_ids可以生成之前的最小token数 | int | 0 | 支持 |
| **生成随机性** | **temperature** | 较低值使模型更确定性<br>较高值使模型更随机<br>0表示贪婪策略，默认为1.0 | float | 1.0 | 支持 |
|  | **top\_p** | (0,1\]范围内，1表示考虑所有token | float | 1.0 | 支持 |
|  | **top\_k** | 考虑的token数。默认-1，表示考虑所有token | int | \-1 | 支持 |
|  | **min\_p** | 考虑token的最小概率（相对于最可能token的概率）<br>范围\[0,1\]，0表示禁用，默认为0 | float | 0.0 | **不支持，运行报错** |
|  | **seed** | 生成使用的随机种子<br>设置为固定值可使生成结果可重现 | int | None | 支持 |
| **重复惩罚** | **presence\_penalty** | \>0鼓励使用新token<br><0鼓励重复token | float | 0.0 | 支持 |
|  | **frequency\_penalty** | \>0鼓励使用新token<br><0鼓励重复token | float | 0.0 | 支持 |
|  | **repetition\_penalty** | \>1鼓励使用新token<br><1鼓励重复token | float | 1.0 | 支持 |
| **内容约束与停止条件** | **stop** | 生成这些字符串时停止生成<br>返回的输出不包含停止字符串 | str\|List\[str\] | None | 支持 |
|  | **stop\_token\_ids** | 生成这些token时停止生成<br>返回的输出包含停止token（除非是特殊token） | List\[int\] | None | 支持 |
|  | **bad\_words** | 不允许生成的单词列表 | List\[str\] | None | 支持 |
|  | **allowed\_token\_ids** | 构建只保留给定token id分数的logits处理器 | List\[int\] | None | 支持 |
|  | **ignore\_eos** | 是否忽略EOS token并继续生成 | bool | False | 支持 |
| **ppl** | **logprobs** | 返回每个输出token的log概率数量<br>默认为None，表示不返回概率<br>非None时返回指定数量的最可能token的log概率 | int | None | 支持 |
|  | **prompt\_logprobs** | 返回每个提示token的log概率数量 | int | None | 支持 |
| **输出格式控制** | **detokenize** | 是否对输出进行detokenize处理 | bool | True | 支持 |
|  | **skip\_special\_tokens** | 是否跳过输出中的特殊token | bool | True | 支持 |
|  | **spaces\_between\_special\_tokens** | 是否在特殊token之间添加空格 | bool | True | 支持 |
|  | **include\_stop\_str\_in\_output** | 是否在输出文本中包含停止字符串 | bool |  | 支持 |
| **高级选项与扩展** | **logits\_processors** | 基于已生成token修改logits的函数列表 | Any | None | 需实现 `LogitsProcessor` 接口，用于复杂约束（如语法检查、动态筛选）。 |
|  | **truncate\_prompt\_tokens** | 如果设置为整数k，只使用提示的最后k个token | int | None |  |
|  | **guided\_decoding** | 根据这些参数构建引导解码logits处理器 | GuidedDecodingParams | None | 支持 JSON 模式、正则表达式、枚举选项等，需配合 `GuidedDecodingParams` 使用。 |
|  | **logit\_bias** | 配合**guided\_decoding**一起构建应用这些logit偏置的logits处理器 | Dict\[int, float\] | None |  |
|  | **extra\_args** | 自定义附加参数（传递给自定义采样器，框架内部不使用）。 | Dict | None |  |


## examples 环境要求

- Python 3.12
- torch==2.7.0+cpu
- vLLM == 0.9.2+cpu(建议使用 harbor.vastaitech.com/ai_deliver/vllm_vacc:latest 镜像)
- 支持VA16/VA1L/VA10L

---

## examples 使用方式

1. 启动容器
```bash
sudo docker run --ipc=host --rm -it --shm-size=256g  \
--network host --privileged \
-v /weights/:/weights/ \
--name=test harbor.vastaitech.com/ai_deliver/vllm_vacc:latest bash
```

>注意: /weights 为模型权重路径

2. 进入容器后可按照下述各 examples 运行方式启动，详细内容可参考各examples 的详细说明

## 离线推理
[离线推理](./offline_inference.md)

## 在线服务
[在线服务](./online_serving.md)

## 工具调用示例
[工具调用示例](./parsing_tool_calls.md)

## 结构化/JSON输出
[JSON输出](./json_output.md)

## 思考与非思考模式
[思考与非思考模式](./thinking&non-thinking_modes.md)
