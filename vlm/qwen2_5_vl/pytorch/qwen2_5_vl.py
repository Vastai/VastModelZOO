# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

import time

from importlib.util import find_spec
if find_spec("torch_vacc"):
    # pip install torch_vacc-1.1-100-cp38-cp38-linux_x86_64.whl
    import torch_vacc
    import torch_vacc.contrib.transfer_to_vacc

t0 = time.time()


# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "vamc_result/weights/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    # device_map="auto",
    # device_map="cuda:1",
    device_map="vacc:1",

)
processor = AutoProcessor.from_pretrained("vamc_result/weights/Qwen2.5-VL-7B-Instruct")

# Image
# url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image = Image.open('imgs/arch.png')

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]


# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

inputs = processor(
    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
)
# inputs = inputs.to("cuda")
inputs = inputs.to("vacc")

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(output_text)

print('infer time(s): ', time.time() - t0)

'''
torch                  2.1.0
torch_vacc             1.1
qwen-vl-utils          0.0.8
transformers           4.45.0
'''
