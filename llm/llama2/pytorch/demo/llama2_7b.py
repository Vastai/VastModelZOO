# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
import torch
import transformers

from importlib.util import find_spec
if find_spec("torch_vacc"):
    import torch_vacc
    import torch_vacc.contrib.transfer_to_vacc


t0 = time.time()

model_id = "Llama-2-7b-chat-hf"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="vacc:1",
    # device_map="cuda:1",
    # device_map="auto",

)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
print('infer time(s): ', time.time() - t0)
