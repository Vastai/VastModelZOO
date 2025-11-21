# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import time
import json
import torch
from transformers import AutoModelForCausalLM, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from safetensors.torch import load_file, save_file


def extract_and_chunk_llm_weights(
    vl_model_path="Qwen/Qwen2-VL-7B-Instruct",
    llm_output_dir="qwen2_7b_llm_chunks",
    max_size_gb=3.5
):
    # 创建输出目录
    os.makedirs(llm_output_dir, exist_ok=True)

    # 加载完整 VL 模型
    vl_model = Qwen2VLForConditionalGeneration.from_pretrained(vl_model_path, torch_dtype="auto", device_map="cpu")

    llm_weights = {
        k: v for k, v in vl_model.state_dict().items() if not k.startswith("visual.")
    }

    for k, v in llm_weights.items():
        if k == "lm_head.weight" or k == "model.embed_tokens.weight":
            llm_weights[k] = v.clone()

    # 分块保存逻辑
    index_data = {
        "metadata": {
            "total_size": sum(
                t.numel() * t.element_size() for t in llm_weights.values()
            )
        },
        "weight_map": {},
    }

    
    current_chunk = {}
    current_size = 0
    chunk_idx = 0
    max_size_bytes = max_size_gb * 1024 ** 3

    for key, tensor in llm_weights.items():
        tensor_size = tensor.numel() * tensor.element_size()
        
        # 如果当前块已满，保存并重置
        if current_size + tensor_size > max_size_bytes and current_chunk:
            chunk_path = f"model-{chunk_idx:04d}.safetensors"
            save_file(current_chunk, os.path.join(llm_output_dir, chunk_path))
            
            # 更新索引
            for k in current_chunk:
                index_data["weight_map"][k] = chunk_path
            
            chunk_idx += 1
            current_chunk = {}
            current_size = 0
        
        current_chunk[key] = tensor
        current_size += tensor_size

    # 保存最后一个分块
    if current_chunk:
        chunk_path = f"model-{chunk_idx:04d}.safetensors"
        save_file(current_chunk, os.path.join(llm_output_dir, chunk_path))
        for k in current_chunk:
            index_data["weight_map"][k] = chunk_path

    # 保存索引文件
    with open(os.path.join(llm_output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"LLM weights saved to {llm_output_dir} with {chunk_idx + 1} chunks.")


if __name__ == "__main__":

    vl_dir = "/path/to/Qwen2-VL-7B-Instruct-GPTQ-Int4"
    vl_llm_dir = "/path/to/Qwen2-VL-7B-Instruct-GPTQ-Int4-llm"
    
    extract_and_chunk_llm_weights(
        vl_model_path=vl_dir,
        llm_output_dir=vl_llm_dir,
        max_size_gb=3.5)
    



    # t0 = time.time()
    # model = AutoModelForCausalLM.from_pretrained(
    #     vl_llm_dir,
    #     device_map="cpu",
    #     torch_dtype=torch.float16
    # )
    # model = model.to("cpu")

    # tokenizer = AutoTokenizer.from_pretrained(vl_llm_dir)

    # input_text = "介绍一下深圳有哪些好玩的地方"
    # inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    # outputs = model.generate(**inputs, max_new_tokens=100)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # print(time.time() - t0)