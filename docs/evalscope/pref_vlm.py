# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

# 参数说明：https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html

task_cfg = Arguments(
    parallel=[1],
    number=[10],
    model='qwen2_vl_7b_llm_28layer_2048_4096_tp4_int8',
    url='http://0.0.0.0:9900/v1/chat/completions',
    api='openai',
    dataset='random_vl',
    min_tokens=128,
    max_tokens=128,
    prefix_length=0,
    min_prompt_length=128,
    max_prompt_length=128,
    image_width=512,
    image_height=512,
    image_format='RGB',
    image_num=1,
    tokenizer_path='ai300/qwen2_vl_7b_llm_28layer_2048_4096_tp4_int8/tokenizer',
    extra_args={'ignore_eos': True}
)
results = run_perf_benchmark(task_cfg)

'''
vastgenx serve --model ai300/qwen2_vl_7b_llm_28layer_2048_4096_tp4_int8 \
--vit_model ai300/qwen2_vl_7b_visual_32layer_5120_fp16 \
--port 9900 \
--llm_devices "[0,1,2,3]" \
--vit_devices "[4]" \
--min_pixels 78400 \
--max_pixels 921600

python VastModelZOO/docs/vastgenx/evalscope/pref_vlm.py
'''