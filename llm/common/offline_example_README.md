#  DeepSeek/Qwen3 离线推理example使用说明

## 简介

`offline_example.py` 是一个用于 **Qwen3 / DeepSeek** 系列模型的 **离线批量推理（offline batch inference）** 的 Python 脚本。  
支持本地模型路径或 HuggingFace 模型路径，支持多推理卡并行推理，适用于无网络环境的部署场景。


---

## 环境要求

- Python 3.12
- torch==2.7.0+cpu
- vLLM == 0.9.2+cpu（建议使用harbor.vastaitech.com/ai_deliver/vllm_vacc:AI3.0_SP9_0811 镜像）
- 支持VA16/VA1L/VA10L

---


## 使用方式

### 1. 查看帮助

```bash
python offline_example.py -h
```

### 2. 本地模型推理示例

```bash
python offline_example.py \
  --model-name /FS03/wyl_data/workspace/weights/Qwen3-30B-A3B-FP8/ \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-tokens 512
```

### 3. HuggingFace 模型路径示例

```bash
python offline_example.py \
  --model-name Qwen/Qwen3-30B-A3B-FP8 \
  --tensor-parallel-size 2 \
  --max-tokens 256
```

---

## 参数说明

| 参数名                     | 说明                                      | 示例值                     |
|--------------------------|-------------------------------------------|----------------------------|
| `--model-name`           | 本地或 HuggingFace 模型路径               | `Qwen/Qwen3-30B-A3B-FP8`   |
| `--tensor-parallel-size` | 张量并行使用的 GPU 数量                   | `4`                        |
| `--max-model-len`        | 输入+输出的最大 token 数                  | `4096`                     |
| `--temperature`          | 控制生成多样性（0~1）                     | `0.7`                      |
| `--top-p`                | 控制 nucleus sampling（0~1）              | `0.9`                      |
| `--max-tokens`           | 每条 prompt 最多生成的 token 数           | `512`                      |

---