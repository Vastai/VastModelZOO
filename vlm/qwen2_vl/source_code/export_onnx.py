# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import sys
# https://github.com/QwenLM/Qwen2-VL/tree/main/qwen-vl-utils
sys.path.append("./code/qwen_vl_build/Qwen2-VL/qwen-vl-utils/src")
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
from pathlib import Path
import os
import onnx
from onnxsim import simplify
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_path = "./code/qwen_vl_build/Qwen2-VL-7B-Instruct"

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float32, device_map="cpu"
)
# print(model.config)

# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "./code/qwen_vl_build/Qwen2-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# # Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to("cpu")
# pixel_values = inputs["pixel_values"]
# image_grid_thw = inputs["image_grid_thw"]

def export_onnx_visual(seq_length):
    # export model.visual onnx
    pixel_values = torch.randn(size=[seq_length, 1176])
    cos =  torch.randn(size=[seq_length, 80])
    sin =  torch.randn(size=[seq_length, 80])
    # attention_mask = torch.zeros([1, seq_length, seq_length], device = pixel_values.device, dtype=torch.bool)
    # attention_mask[:, :seq_length, :seq_length] = True

    attention_mask=torch.ones([1, seq_length, seq_length])

    save_path =  f'./qwen2_vl_7b_onnx_{model.config.vision_config.depth}block/'
    Path(f'{save_path}').mkdir(parents=True, exist_ok=True)
    onnx_fpath = f'{save_path}/qwen2_vl_7b_visual.onnx'

    save_path_sim =  f'./qwen2_vl_7b_visual_onnx_{model.config.vision_config.depth}block_{seq_length}_sim/'
    Path(f'{save_path_sim}').mkdir(parents=True, exist_ok=True)
    onnx_fpath_sim = f'{save_path_sim}/qwen2_vl_7b_visual_{seq_length}_sim_newjz.onnx'
    os.system(f"rm -rf {save_path_sim}/*")

    model.visual.forward = model.visual.forward_vacc
    torch.onnx.export(
        model.visual,
        (
            pixel_values,
            attention_mask,
            cos,
            sin,
        ),
        onnx_fpath,
        input_names=["pixel_values", 'attention_mask', 'cos', 'sin'],
        output_names=["image_embeds"],
    )

    onnx_model = onnx.load_model(onnx_fpath, load_external_data=True)
    model_simp, check = simplify(onnx_model)
    os.system(f"rm -rf {save_path}")
    onnx.save(model_simp, onnx_fpath_sim, save_as_external_data=True, all_tensors_to_one_file=True)


# export model.llm onnx
config = model.config
def export_onnx_iter0(input_seq_len = 2048):
    # iter0
    save_path =  f'./qwen2_vl_7b_onnx_{model.config.num_hidden_layers}block_iter0/'
    Path(f'{save_path}').mkdir(parents=True, exist_ok=True)
    onnx_fpath = f'{save_path}/qwen2_vl_7b_llm_iter0.onnx'

    save_path_sim =  f'./qwen2_vl_7b_onnx_{model.config.num_hidden_layers}block_{input_seq_len}_iter0_sim/'
    Path(f'{save_path_sim}').mkdir(parents=True, exist_ok=True)
    onnx_fpath_sim = f'{save_path_sim}/qwen2_vl_7b_llm_{input_seq_len}_iter0_sim.onnx'
    os.system(f"rm -rf {save_path_sim}/*")

    input_ids=None
    attention_mask=None
    cos=torch.randn(size=[1, 1, input_seq_len, 128])
    sin=torch.randn(size=[1, 1, input_seq_len, 128])
    past_key_values=None
    inputs_embeds=torch.randn(size=[1, input_seq_len, config.hidden_size])
    use_cache=True
    output_attentions=False
    output_hidden_states=False
    return_dict=True

    inputs = (
        input_ids,
        attention_mask,
        cos,
        sin, 
        past_key_values,
        inputs_embeds,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
    )

    # for the first iteration with no history KV cache
    model.model.forward = model.model.forward_vacc
    torch.onnx.export(
        model.model,
        inputs,
        onnx_fpath,
        opset_version=14,
    )

    onnx_model = onnx.load_model(onnx_fpath, load_external_data=True)
    model_simp, check = simplify(onnx_model)
    os.system(f"rm -rf {save_path}")
    onnx.save(model_simp, onnx_fpath_sim, save_as_external_data=True, all_tensors_to_one_file=True)


def export_onnx_iter1(input_seq_len = 2048):
    # iter0
    save_path =  f'./qwen2_vl_7b_onnx_{model.config.num_hidden_layers}block_iter1/'
    Path(f'{save_path}').mkdir(parents=True, exist_ok=True)
    onnx_fpath = f'{save_path}/qwen2_vl_7b_llm_iter1.onnx'

    save_path_sim =  f'./qwen2_vl_7b_onnx_{model.config.num_hidden_layers}block_{input_seq_len}_iter1_sim/'
    Path(f'{save_path_sim}').mkdir(parents=True, exist_ok=True)
    onnx_fpath_sim = f'{save_path_sim}/qwen2_vl_7b_llm_{input_seq_len}_iter1_sim.onnx'
    os.system(f"rm -rf {save_path_sim}/*")

    input_ids=None
    cos=torch.randn(size=[1, 1, 1, 128])
    sin=torch.randn(size=[1, 1, 1, 128])
    attention_mask=torch.ones([1, input_seq_len], dtype=torch.int64)
    past_key_values=torch.randn(config.num_hidden_layers, 2, 1, config.num_key_value_heads, input_seq_len - 1, 128, dtype=torch.float32)
    inputs_embeds=torch.randn(size=[1, 1, config.hidden_size])
    use_cache=True
    output_attentions=False
    output_hidden_states=False
    return_dict=True

    inputs = (
        input_ids,
        attention_mask,
        cos,
        sin, 
        past_key_values,
        inputs_embeds,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
    )

    model.model.forward = model.model.forward_vacc
    torch.onnx.export(
        model.model,
        inputs,
        onnx_fpath,
        opset_version=14,
    )

    onnx_model = onnx.load_model(onnx_fpath, load_external_data=True)
    model_simp, check = simplify(onnx_model)
    os.system(f"rm -rf {save_path}")
    onnx.save(model_simp, onnx_fpath_sim, save_as_external_data=True, all_tensors_to_one_file=True)

export_onnx_visual(seq_length=4096)
# export_onnx_iter0(2048)
# export_onnx_iter1(2048)
