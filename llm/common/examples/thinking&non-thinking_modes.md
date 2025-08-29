
# Qwen3 思考与非思考模式配置指南


## 前提条件
Qwen3 模型在回复前会进行思考。此行为可通过以下方式控制：
- **硬开关**：完全禁用思考
参考vllm 运行命令:
```bash
vllm serve /weights/Qwen3-30B-A3B-FP8 \
--trust-remote-code \
--tensor-parallel-size 4 \
--max-model-len 32768 \
--enforce-eager \
--host 0.0.0.0 \
--port 8001 \
--chat-template /workspace/qwen3_nonthinking.jinja
```
该聊天模板会阻止模型生成思考内容，即使用户通过 /think 指示模型这样做。


- **软开关**：模型遵循用户指令决定是否思考
参考 vllm 运行命令：
```bash
vllm serve /weights/Qwen3-30B-A3B-FP8 \
--trust-remote-code \
--tensor-parallel-size 4 \
--max-model-len 32768 \
--enforce-eager \
--host 0.0.0.0 \
--port 8001 \
--reasoning-parser qwen3
```
---

## thinking_non-thinking_models.py 命令参数说明
```bash
options:
  -h, --help            show this help message and exit
  --host HOST           Host address (default: 127.0.0.1)
  --port PORT           Port number (default: 8000)
  --model-name MODEL_NAME
                        Model name (default: DeepSeek-V3)
  --disable-thinking    if set, will disable thinking
```

## thinking_non-thinking_models.py 运行示例
```bash
#禁用思考模式
python3 thinking_non-thinking_modes.py \
--host 10.24.73.25 \
--port 8001 \
--model-name Qwen3  \
--disable-thinking

#启用思考模式
python3 thinking_non-thinking_modes.py \
--host 10.24.73.25 \
--port 8001 \
--model-name Qwen3

```

## thinking_non-thinking_models.py 运行结果示例
```bash
#禁用思考模式结果示例
Chat response: ChatCompletion(id='chatcmpl-5596fcab734444059a412a6461cd1326', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='\n\nLarge language models (LLMs) are advanced artificial intelligence systems designed to understand and generate human-like text. They are trained on vast amounts of data from the internet, allowing them to perform a wide range of tasks such as answering questions, writing essays, coding, and even engaging in conversation. These models use deep learning techniques, particularly transformer architectures, to process and generate language with remarkable fluency and context awareness. LLMs have become a cornerstone of modern AI, enabling significant advancements in natural language processing and human-computer interaction.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[], reasoning_content='\n\n'), stop_reason=None)], created=1755689331, model='Qwen3', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=111, prompt_tokens=21, total_tokens=132, completion_tokens_details=None, prompt_tokens_details=None), prompt_logprobs=None, kv_transfer_params=None)

#启用思考模式结果示例

Chat response: ChatCompletion(id='chatcmpl-717355db3cf0490a9bc4d542c7d87e0d', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='\n\nLarge language models (LLMs) are advanced artificial intelligence systems designed to understand and generate human-like text by analyzing vast amounts of data. Trained on extensive datasets, these models use deep learning techniques, particularly neural networks, to recognize patterns, context, and relationships in language. They can perform tasks like answering questions, writing essays, coding, and engaging in conversations. LLMs excel at handling complex linguistic tasks but rely on statistical patterns rather than true comprehension, and their outputs depend heavily on the quality and biases of their training data. They are widely used in applications such as virtual assistants, content creation, and customer service, though they require careful oversight to ensure accuracy and ethical use.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[], reasoning_content="\nOkay, the user is asking for a short introduction to large language models. Let me start by recalling what I know. Large language models, or LLMs, are a type of AI that processes and generates human-like text. They're trained on vast amounts of data, which allows them to understand and produce text in multiple languages.\n\nI should mention their training process. They use deep learning, specifically neural networks, and are trained on huge datasets. The key here is the scale—both the amount of data and the number of parameters in the model. Maybe I should explain that the more data and parameters, the better the model can understand context and generate coherent responses.\n\nApplications are important too. They can answer questions, write essays, code, and even engage in conversations. But I should also touch on their limitations. For example, they might not always be accurate, can have biases from the training data, and don't truly understand the content they generate, just patterns.\n\nI need to keep it concise. Avoid jargon where possible, but some terms like neural networks and parameters are necessary. Maybe start with a definition, then how they work, their uses, and their limitations. Make sure it's clear and easy to understand. Let me check if there's anything else important. Oh, maybe mention that they're a subset of machine learning and part of the broader AI field. Also, note that they're used in various industries like customer service, content creation, etc. But keep it short. Alright, time to put it all together in a coherent way.\n"), stop_reason=None)], created=1755689303, model='Qwen3', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=456, prompt_tokens=19, total_tokens=475, completion_tokens_details=None, prompt_tokens_details=None), prompt_logprobs=None, kv_transfer_params=None)
```



## curl 命令调用示例
```bash
#Qwen3 禁用思考模式
curl http://10.24.73.5:8001/v1/chat/completions -H "Content-Type: application/json" -d '{
	"model": "Qwen3",
	"messages": [
		{
			"role": "user",
			"content": "Give me a short introduction to large language models."
		}
	],
	"temperature": 0.7,
	"top_p": 0.8,
	"max_tokens": 8192,
	"chat_template_kwargs": {
		"enable_thinking": false
	}
}'

#DeepSeek-V3.1 开启思考模式
curl http://10.24.73.5:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
	"model": "DeepSeek-V3",
	"messages": [
		{
			"role": "user",
			"content": "Give me a short introduction to large language models."
		}
	],
	"temperature": 0.7,
	"top_p": 0.8,
	"max_tokens": 8192,
	"chat_template_kwargs": {
		"thinking": true
	}
}'
```

## curl 命令调用结果
```bash
#Qwen3 禁用思考模式
{"id":"chatcmpl-79f50ad351054772a79ab388104f2031","object":"chat.completion","created":1755689585,"model":"Qwen3","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":null,"content":"Large language models (LLMs) are advanced artificial intelligence systems designed to understand and generate human-like text. Trained on vast amounts of data from the internet, they can perform a wide range of tasks, such as answering questions, writing essays, coding, and even engaging in conversation. These models use deep learning techniques, particularly transformer architectures, to process and generate language with remarkable fluency and context awareness. LLMs have become foundational tools in many areas of AI, enabling more natural and effective interactions between humans and machines.","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":22,"total_tokens":128,"completion_tokens":106,"prompt_tokens_details":null},"prompt_logprobs":null,"kv_transfer_params":null}
```

```bash
#DeepSeek-V3.1 开启思考模式
{"id":"chatcmpl-81205c3d6b434a398f6373d3b576eafa","object":"chat.completion","created":1756276026,"model":"DeepSeek-V3","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":null,"content":"Hmm, the user wants a short introduction to large language models. I need to keep it concise but informative. I can start with a simple analogy to make it accessible, then briefly explain the core mechanism—transformer architecture and self-attention. I should also highlight key features like scale, pretraining, and fine-tuning, but avoid technical details. Ending with common use cases and a note on ethical considerations would round it out. Let me structure it so it’s easy to follow but still substantive.</think>Of course. Here is a short introduction to large language models.\n\n### Large Language Models (LLMs): A Brief Overview\n\nA large language model (LLM) is a type of artificial intelligence that has been trained on a massive amount of text data. Think of it as a powerful, automated autocomplete system.\n\nIts core function is to **predict the next most likely word in a sequence**. By doing this repeatedly, it can generate entire sentences, paragraphs, and even documents that are coherent and contextually relevant.\n\n**How do they work?**\n*   **Training:** They learn by analyzing billions of sentences from books, articles, and websites. This process allows them to learn grammar, facts, reasoning abilities, and even some nuances of human communication.\n*   **The \"Transformer ArchitectureTransformer Architecture:** Most LLMs are based on a breakthrough called the \"Transformer\" architecture. This allows them to understand the relationships between words in a sentence, no matter how far apart they are, making their outputs more accurate and sensible.\n\n**What makes them \"large\"?**\nThe \"large\" refers to the immense scale of two things:\n1.  The size of their **training data** (often encompassing a significant portion of the internet).\n2.  The number of **parameters** (the internal settings the model adjusts during training, which can number in the billions or trillions).\n\n**What can they do?**\nLLMs power a wide range of applications, including:\n*   Chatbots and virtual assistants (like ChatGPT)\n*   Translating languages\n*   Summarizing long documents\n*   Writing code and debugging\n*   Drafting emails and content\n\nIn essence, LLMs are powerful pattern-matching systems that generate human-like text, making them a transformative technology for human-computer interaction.","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":14,"total_tokens":481,"completion_tokens":467,"prompt_tokens_details":null},"prompt_logprobs":null,"kv_transfer_params":null}
```