# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import vllm_vacc
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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
        "今天天气真好，",
        "人工智能是",
        "法国的首都是"
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"初始化模型: {args.model_name} , TP={args.tensor_parallel_size}")
    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len
    )

    
    input_ids = []
    for text in prompts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        input_ids.append(tokens)

    print("Input tokens:", input_ids)

    print("开始批量推理...")
    start = time.time()
    outputs = llm.generate(prompt_token_ids=input_ids, 
                           sampling_params=sampling_params)
    duration = time.time() - start

    print(f"\n完成! 总耗时: {duration:.2f}s\n")
    for i, output in enumerate(outputs):
        print(f"【Prompt {i+1}】")
        print(f"输入: {output.prompt_token_ids}")
        print(f"输出: {output.outputs[0].text.strip()}")
        print("-" * 80)

    del llm
    gc.collect()