
# 结构化/JSON输出 使用说明

## 前提条件
已启动在线 vllm 服务


## json_output.py 命令参数说明

```bash
python json_output.py -h

options:
  -h, --help            show this help message and exit
  --host HOST           Host address (default: 127.0.0.1)
  --port PORT           Port number (default: 8000)
  --model-name MODEL_NAME
                        Model name (default: DeepSeek-V3)
```


## json_output.py 运行示例

```bash
python json_output.py \
--host 10.24.73.25 \
--port 8001 \
--model-name Qwen3
```

## json_output.py 运行结果示例
```bash
{
  "programming_languages": [
    {
      "name": "Python",
      "use": "数据科学、机器学习、Web 开发、自动化脚本"
    },
    {
      "name": "JavaScript",
      "use": "前端开发、Web 应用、服务器端开发（Node.js）、移动应用开发"
    },
    {
      "name": "Java",
      "use": "企业级应用、Android 应用开发、大型系统架构"
    }
  ]
}
```