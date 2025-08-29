# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

# 参数说明：https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html

task_cfg = Arguments(
    parallel=[1],
    number=[10],
    model='Qwen2.5-7B-Instruct-int8-tp8-2048-4096',
    url='http://0.0.0.0:9900/v1/chat/completions',
    api='openai',
    dataset='random',
    min_tokens=1024,
    max_tokens=1024,
    prefix_length=0,
    min_prompt_length=1024,
    max_prompt_length=1024,
    tokenizer_path='ai300/Qwen2.5-7B-Instruct-int8-tp8-2048-4096/tokenizer',
    extra_args={'ignore_eos': True}
)
results = run_perf_benchmark(task_cfg)

'''
vastgenx serve --model ai300/Qwen2.5-7B-Instruct-int8-tp8-2048-4096 \
--port 9900 \
--llm_devices "[0,1,2,3,4,5,6,7]"

python VastModelZOO/docs/vastgenx/evalscope/pref_llm.py
'''