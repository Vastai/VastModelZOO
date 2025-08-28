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
    "vamc_result/weights/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    # device_map="auto",
    # device_map="cuda:1",
    device_map="vacc:1",

)
processor = AutoProcessor.from_pretrained("vamc_result/weights/Qwen2-VL-2B-Instruct")

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


'''
VA16L, 4*4*32GB

Qwen2-VL-2B-Instruct

device_map="vacc:1"
+=============================================================================================================+
|  0/0    23795              python                                                0.00B                      |
|  0/1    23795              python                                               6.91GB                      |
+-------------------------------------------------------------------------------------------------------------+
['The image appears to be a placeholder or a graphic design used for a website or a presentation. It features a series of geometric shapes arranged in a grid pattern. The shapes are:\n\n1. A square\n2. A circle\n3. A triangle\n4. A parallelogram\n\nEach shape is filled with a solid color, and the background is a muted, neutral color, possibly brown or beige. The shapes are evenly spaced and aligned horizontally and vertically. The overall design is simple and minimalistic, with a clean and professional look.']
infer time(s):  61.76532435417175


device_map="auto"
+=============================================================================================================+
|  0/0    21735              python                                              96.00MB                      |
|  0/1    21735              python                                             704.00MB                      |
|  0/2    21735              python                                             544.00MB                      |
|  0/3    21735              python                                             544.00MB                      |
|  1/0    21735              python                                             208.00MB                      |
|  1/1    21735              python                                              16.00MB                      |
|  1/2    21735              python                                              16.00MB                      |
|  1/3    21735              python                                              16.00MB                      |
|  2/0    21735              python                                              16.00MB                      |
|  2/1    21735              python                                              16.00MB                      |
|  2/2    21735              python                                              16.00MB                      |
|  2/3    21735              python                                              16.00MB                      |
|  3/0    21735              python                                              16.00MB                      |
|  3/1    21735              python                                              16.00MB                      |
|  3/2    21735              python                                              16.00MB                      |
|  3/3    21735              python                                               3.52GB                      |
+-------------------------------------------------------------------------------------------------------------+

Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.46s/it]
[2025-01-17 15:27:33.258][21735][warning][utils.cc:616, TryAdaptOpOnVacc] [Dtype Conversion] op_binary_lt : {Long,Long} -> {Int,Int}
** Python Context:
File "./miniconda3/envs/vamc/lib/python3.8/site-packages/transformers/generation/utils.py", line 1687, in _prepare_special_tokens
  torch.is_floating_point(eos_token_tensor) or (eos_token_tensor < 0).any()
[2025-01-17 15:27:33.267][21735][warning][index_ops.cc:464, _index_put_impl_] The operator 'Tensor{shape=[3, 3602], type=long int, device=vacc:0, contiguous, nbytes=86448, offset=0} [None, Tensor{shape=[3602], type=bool, device=vacc:0, contiguous, nbytes=3602, offset=0}, ] <- Tensor{shape=[3, 3602], type=long int, device=vacc:0, contiguous, nbytes=86448, offset=0} accumulate: 0' is not currently supported on the vacc backend, fallback to CPU.
** Python Context:
File "./miniconda3/envs/vamc/lib/python3.8/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py", line 1577, in get_rope_index
  position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
Traceback (most recent call last):
  File "vamc_results/torch_vacc_infer_vlm.py", line 58, in <module>
    output_ids = model.generate(**inputs, max_new_tokens=128)
  File "./miniconda3/envs/vamc/lib/python3.8/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "./miniconda3/envs/vamc/lib/python3.8/site-packages/transformers/generation/utils.py", line 2048, in generate
    result = self._sample(
  File "./miniconda3/envs/vamc/lib/python3.8/site-packages/transformers/generation/utils.py", line 3008, in _sample
    outputs = self(**model_inputs, return_dict=True)
  File "./miniconda3/envs/vamc/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "./miniconda3/envs/vamc/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "./miniconda3/envs/vamc/lib/python3.8/site-packages/accelerate/hooks.py", line 170, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "./miniconda3/envs/vamc/lib/python3.8/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py", line 1691, in forward
    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
  File "./miniconda3/envs/vamc/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "./miniconda3/envs/vamc/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "./miniconda3/envs/vamc/lib/python3.8/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py", line 1120, in forward
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
  File "./miniconda3/envs/vamc/lib/python3.8/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py", line 1092, in rot_pos_emb
    hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
  File "./miniconda3/envs/vamc/lib/python3.8/site-packages/torch_vacc/contrib/transfer_to_vacc.py", line 117, in decorated
    return fn(*args, **kwargs)
RuntimeError: upper bound and larger bound inconsistent with step sign
'''


'''
NV-H800, 8*80GB

Qwen2-VL-2B-Instruct

device_map="cuda:1"
|=======================================================================================|
|    1   N/A  N/A   2988605      C   python                                    19604MiB |
+---------------------------------------------------------------------------------------+
['The image depicts a serene beach scene with a woman and her dog. The woman is sitting on the sand, wearing a plaid shirt and black pants, and appears to be smiling. She is holding up her hand in a high-five gesture towards the dog, which is also sitting on the sand. The dog has a harness on, and its front paws are raised in a playful manner. The background shows the ocean with gentle waves, and the sky is clear with a soft glow from the setting or rising sun, casting a warm light over the entire scene. The overall atmosphere is peaceful and joyful.']
infer time(s):  10.888946294784546


device_map="auto"
|=======================================================================================|
|    0   N/A  N/A   2983451      C   python                                    15334MiB |
|    1   N/A  N/A   2983451      C   python                                    15438MiB |
|    2   N/A  N/A   2983451      C   python                                    15104MiB |
|    3   N/A  N/A   2983451      C   python                                     5016MiB |
|    4   N/A  N/A   2983451      C   python                                     1882MiB |
|    5   N/A  N/A   2983451      C   python                                     1882MiB |
|    6   N/A  N/A   2983451      C   python                                     1882MiB |
|    7   N/A  N/A   2983451      C   python                                     1664MiB |
+---------------------------------------------------------------------------------------+

['The image depicts a serene beach scene with a woman and her dog. The woman is sitting on the sand, wearing a plaid shirt and black pants, and appears to be smiling. She is holding up her hand in a high-five gesture towards the dog, which is also sitting on the sand. The dog has a harness on, and its front paws are raised in a playful manner. The background shows the ocean with gentle waves, and the sky is clear with a soft glow from the setting or rising sun, casting a warm light over the entire scene. The overall atmosphere is peaceful and joyful.']
infer time(s):  23.909090518951416
'''