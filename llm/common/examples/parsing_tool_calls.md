
# 工具调用示例


## 前提条件
已启动在线 vllm 服务,并设置--enable-auto-tool-choice --tool-call-parser 参数。

以 Qwen3-30B-A3B-FP8 为例, 启动参数:
```bash
vllm serve /weights/Qwen3-30B-A3B-FP8 \
--trust-remote-code \
--tensor-parallel-size 4 \
--max-model-len 32768 \
--enforce-eager \
--host 0.0.0.0 \
--port 8001 \
--enable-auto-tool-choice \
--tool-call-parser hermes
```

## parsing_tool_calls.py 命令参数说明
```bash
python parsing_tool_calls.py -h

options:
  -h, --help            show this help message and exit
  --host HOST           Host address (default: 127.0.0.1)
  --port PORT           Port number (default: 8000)
  --model-name MODEL_NAME
                        Model name (default: DeepSeek-V3)
```


## parsing_tool_calls.py 运行示例

```bash
python parsing_tool_calls.py \
--host 10.24.73.25 \
--port 8001 \
--model-name Qwen3
```

## parsing_tool_calls 运行结果示例
```bash
User>    How's the weather in Hangzhou?
tool:ChatCompletionMessageToolCall(id='chatcmpl-tool-0b13f9c4b99d484191281a500b3a22f1', function=Function(arguments='{"location": "Hangzhou"}', name='get_weather'), type='function')
Model>   The current weather in Hangzhou is 24℃.
```

## parsing_tool_calls.py 代码解释
这段代码展示了一个与本地部署的AI模型（Qwen3）进行交互，并使用函数调用（tools）获取天气信息的完整流程。

### 第一次调用 `send_messages`
```python
messages = [{"role": "user", "content": "How's the weather in Hangzhou?"}]
message = send_messages(messages)
```

**作用**：处理用户的天气查询请求，识别需要调用工具函数

**处理过程**：
1. 用户询问"杭州天气如何？"
2. 模型分析发现需要调用`get_weather`函数来获取天气信息
3. 模型返回一个包含工具调用请求的响应

**返回结果**： 
- ChatCompletionMessageToolCall(id='chatcmpl-tool-0b13f9c4b99d484191281a500b3a22f1', function=Function(arguments='{"location": "Hangzhou"}', name='get_weather'), type='function')

>`message.tool_calls[0]` 包含调用`get_weather`函数的请求：

> 参数：`{"location": "Hangzhou"}`

### 中间处理
```python
messages.append(message)  # 将模型的工具调用响应加入对话历史
messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24℃"})  # 模拟工具返回结果
```

**作用**：模拟工具执行并返回结果，构建完整的对话上下文

### 第二次调用 `send_messages`
```python
message = send_messages(messages)
```

**作用**：基于工具返回的天气信息生成最终回答

**处理过程**：
1. 模型接收到工具返回的"24℃"数据
2. 结合之前的对话上下文，生成友好的天气回答
3. 返回完整的自然语言响应

**预期返回结果**：
类似："The current weather in Hangzhou is 24℃"

### 整体流程总结

| 调用次数 | 输入内容 | 输出内容 | 目的 |
|---------|---------|---------|------|
| 第一次 | 用户天气查询 | 工具调用请求 | 识别需要外部数据，准备调用函数 |
| 第二次 | 工具返回结果+历史 | 最终天气回答 | 整合信息，生成用户友好的响应 |

### 关键技术点

1. **工具调用机制**：模型可以识别何时需要调用外部函数
2. **对话状态管理**：通过维护`messages`数组来保持对话上下文
3. **角色系统**：
   - `user`: 用户输入
   - `assistant`: 模型响应（可能包含工具调用）
   - `tool`: 工具执行结果

这种模式使得AI模型能够与外部系统交互，获取实时数据后再生成最终回答，大大扩展了模型的能力范围。