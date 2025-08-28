# Online Serving使用说明


## 前提条件
请参考[README](./README.md) 中"examples 使用方式"部分内容


### 启动 vllm 服务

```bash
vllm serve /weights/Qwen3-30B-A3B-FP8 \
--trust-remote-code \
--tensor-parallel-size 4 \
--max-model-len 32768 \
--enforce-eager \
--host 0.0.0.0 \
--port 8001
```

| 参数 | 说明 |
| --------------------- | ------------------------------------------------------ | 
| `--model-name`        | 本地或 HuggingFace 模型路径 |
| `--tensor-parallel-size` | 张量并行 数量 |
| `--max-model-len` | 最大 token 长度 |
| `--host` | 监听地址 |
| `--port` | 监听端口 |
| `--enforce-eager`     |  强制使用 Eager 模式       | 
| `--trust-remote-code` |  允许加载远程自定义代码  | 
更详细参数说明，请通过 `vllm serve --help` 查看。
---

## online_serving_cli.py 命令参数说明
```bash
python online_serving.py -h

options:
  -h, --help            show this help message and exit
  --host HOST           Host address (default: 127.0.0.1)
  --port PORT           Port number (default: 8000)
  --model-name MODEL_NAME
                        Model name (default: DeepSeek-V3)
```

## online_serving_cli.py 运行示例
```bash
python online_serving_cli.py \
--host 10.24.73.25 \
--port 8001 \
--model-name Qwen3
```

## online_serving_cli.py 运行结果示例
```bash
1. **Python**  
   - **主要用途**：数据科学、人工智能（AI）、机器学习、Web 开发、自动化脚本、科学计算。  
   - **特点**：语法简洁易读，拥有丰富的第三方库（如 NumPy、Pandas、TensorFlow、Django 等），适合初学者和专业开发者。

2. **JavaScript**  
   - **主要用途**：前端 Web 开发（在浏览器中实现交互功能）、后端开发（通过 Node.js）、移动应用开发（如 React Native）。  
   - **特点**：是唯一一种在浏览器中原生支持的编程语言，广泛用于构建动态网页和实时应用。

3. **Java**  
   - **主要用途**：企业级应用开发（如银行系统、大型电商平台）、Android 应用开发、大型分布式系统。  
   - **特点**：跨平台（“一次编写，到处运行”）、稳定性高、面向对象，广泛应用于工业级软件开发。

这三种语言在现代软件开发中都占据重要地位，选择哪一种通常取决于具体的应用场景和开发需求。
```



## curl 命令请求方式示例
```bash
curl -X POST http://10.24.73.25:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-abc123" \
  -d '{
    "model": "Qwen3",
    "messages": [
      {"role": "system", "content": "你是一个专业助手"},
      {"role": "user", "content": "列出 3 种编程语言及其主要用途"}
    ],
    "max_tokens": 1000
  }'
```



## curl 命令运行结果示例
```bash
{
	"id": "chatcmpl-3d69d920e5cf49afa23ff342fd6e2ddd",
	"object": "chat.completion",
	"created": 1755679363,
	"model": "Qwen3",
	"choices": [
		{
			"index": 0,
			"message": {
				"role": "assistant",
				"reasoning_content": null,
				"content": "当然，以下是三种常见的编程语言及其主要用途：\n\n1. **Python**  \n   - **主要用途**：数据科学、人工智能（AI）、机器学习、Web开发、自动化脚本、科学计算。  \n   - **特点**：语法简洁易读，拥有丰富的第三方库（如NumPy、Pandas、TensorFlow、Django等），适合初学者和专业开发者。\n\n2. **JavaScript**  \n   - **主要用途**：前端Web开发（网页交互）、后端开发（Node.js）、移动应用开发（React Native）、游戏开发。  \n   - **特点**：浏览器原生支持，是构建动态网页和交互式用户界面的核心语言。\n\n3. **Java**  \n   - **主要用途**：企业级应用开发（如银行系统）、Android应用开发、大型分布式系统、Web后端服务。  \n   - **特点**：跨平台（“一次编写，到处运行”）、强类型、稳定性高，广泛应用于大型系统和企业环境。\n\n这三种语言在现代软件开发中占据重要地位，各有其优势和适用场景。",
				"tool_calls": []
			},
			"logprobs": null,
			"finish_reason": "stop",
			"stop_reason": null
		}
	],
	"usage": {
		"prompt_tokens": 27,
		"total_tokens": 257,
		"completion_tokens": 230,
		"prompt_tokens_details": null
	},
	"prompt_logprobs": null,
	"kv_transfer_params": null
}
```

