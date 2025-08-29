# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import vllm_vacc
from vllm import LLM, SamplingParams
import gc
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3 / DeepSeek 离线批量推理")
    parser.add_argument("--model-name", type=str, required=True,
                        help="本地或 HuggingFace 模型路径")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="张量并行 GPU 数量")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="输入+输出的最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="每条 prompt 最多生成的 token 数")
    args = parser.parse_args()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|endoftext|>", "\n\n\n"]
    )

    prompts = [
        "用Python写一个快速排序函数，并添加详细注释",
        "解释量子计算与传统计算的本质区别",
        "写一段关于人工智能伦理的200字短文",
        "将以下自然语言描述转为SQL查询：查询2023年销售额超过100万的电子产品",
    ]

    print(f"初始化模型: {args.model_name} , TP={args.tensor_parallel_size}")
    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len
    )

    print("开始批量推理...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    duration = time.time() - start

    print(f"\n完成! 总耗时: {duration:.2f}s\n")
    for i, output in enumerate(outputs):
        print(f"【Prompt {i+1}】")
        print(f"输入: {output.prompt}")
        print(f"输出: {output.outputs[0].text.strip()}")
        print("-" * 80)

    del llm
    gc.collect()