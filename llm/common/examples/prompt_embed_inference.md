#  prompt_embed_inference使用说明

## 前提条件
请参考[README](./README.md) 中"examples 使用方式"部分内容


## prompt_embed_inference.py 命令参数说明

```bash
python prompt_embed_inference.py -h

options:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        本地或 HuggingFace 模型路径
  --tensor-parallel-size TENSOR_PARALLEL_SIZE
                        张量并行 GPU 数量
  --max-model-len MAX_MODEL_LEN
                        输入+输出的最大 token 数
  --temperature TEMPERATURE
  --top-p TOP_P
  --max-tokens MAX_TOKENS
                        每条 prompt 最多生成的 token 数
```

## prompt_embed_inference.py 运行示例

```bash
python prompt_embed_inference.py \
  --model-name /weights/Qwen3-30B-A3B-FP8/ \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-tokens 512
```

## prompt_embed_inference.py 运行结果示例
```bash
完成! 总耗时：47.34s
【Prompt 1】
输入: [100644, 104307, 88051, 52801, 3837]
输出: xxxxxxxxxxxxxxxxxxxxxxxxxxxx
--------------------------------------------------------------------------------
【Prompt 2】
输入: [104455, 20412]
输出: xxxxxxxxxxxxxxxxxxxxxxxxxxxx
-------------------------------------------------------------------------------- 【Prompt 3】
输入: 写一段关于人工智能伦理的200字短文
输出: xxxxxxxxxxxxxxxxxxxxxxxxxxxx
--------------------------------------------------------------------------------
【Prompt 4】
输入: [104328, 9370, 59975, 100132]
输出: xxxxxxxxxxxxxxxxxxxxxxxxxxxx
--------------------------------------------------------------------------------
```

