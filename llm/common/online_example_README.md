# DeepSeek/Qwen3 在线推理example使用说明

支持 OpenAI 兼容的 `/v1/chat/completions` 与 `/v1/completions` 接口，可在多推理卡并行推理 DeepSeek/Qwen3模型。

---

## 1. 环境依赖

- Python 3.12
- torch==2.7.0+cpu
- vLLM==0.9.2+cpu（建议使用harbor.vastaitech.com/ai_deliver/vllm_vacc:AI3.0_SP7_0728 镜像）
- 支持VA16/VA1L/VA10L


---

## 2. 启动服务

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

## 3. 调用示例

### 3.1 Chat（对话）接口

```bash
curl -X POST http://<host>:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-abc123" \
  -d '{
    "model": "Qwen3-30B-A3B-FP8",
    "messages": [
      {"role": "system", "content": "你是一个专业助手"},
      {"role": "user", "content": "天津有哪些好玩的地方？"}
    ],
    "max_tokens": 1000
  }'
```

返回字段与 OpenAI 完全一致，示例：

```json
{
  "id": "chatcmpl-1719301234",
  "object": "chat.completion",
  "created": 1719301234,
  "model": "Qwen3-30B-A3B-FP8",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "天津的热门景点包括：\n1. 天津之眼（海河摩天轮）\n2. 古文化街（品尝狗不理包子）\n3. 五大道（欧式建筑群）"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 67,
    "total_tokens": 95
  },
  "latency": "0.982s"
}
```
