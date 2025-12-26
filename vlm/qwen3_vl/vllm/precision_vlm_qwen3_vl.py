# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy


task_cfg = TaskConfig(
    model='Qwen3-VL-30B-A3B-Instruct-FP8',
    # model='Qwen3-VL-30B-A3B-Thinking-FP8',

    api_url='http://10.24.73.28:8000/v1',
    api_key='API_KEY',
    eval_type=EvalType.SERVICE,

    datasets=[
        'gpqa_diamond',      # puretext, knowlegde
        'aime25',            # puretext, math
        'live_code_bench',   # puretext, code

        'math_vista',        # vision, math
        'ocr_bench_v2',      # vision, ocr
        'vstar_bench',       # vision, grounding
        'blink'              # vision, multi-image
    ],
    dataset_args={
        'gpqa_diamond': {"filters": {"remove_until": "</think>"}},
        'math_500': {"filters": {"remove_until": "</think>"}},
        'live_code_bench': {
            'subset_list': ['v6'],
            # 'extra_params': {
            #     'start_date': '2025-02-01',
            #     'end_date': '2025-05-31'
            # },
            "filters": {"remove_until": "</think>"}
        },
        'math_vista': {"filters": {"remove_until": "</think>"}},
        'ocr_bench_v2': {"filters": {"remove_until": "</think>"}},
        'vstar_bench': {"filters": {"remove_until": "</think>"}},
        'blink': {"filters": {"remove_until": "</think>"}},
    },

    eval_batch_size=4,
    repeats=1,

    ## Instruct
    generation_config={
        'max_tokens': 32768,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.7,  # 采样温度 (qwen3-vl 报告推荐值)
        'top_p': 0.8,  # top-p采样 (qwen3-vl 报告推荐值)
        'top_k': 20,  # top-k采样 (qwen3-vl 报告推荐值)
        'repetition_penalty': 1.0,  # 重复惩罚系数
        'presence_penalty': 1.5,  # 新内容惩罚系数
        'n': 1,  # 每个请求产生的回复数量
        'seed': 42
    },
    
    ## Thinking
    # generation_config={
    #     'max_tokens': 40960,  # 最大生成token数，建议设置为较大值避免输出截断
    #     'temperature': 0.6,  # 采样温度 (qwen 报告推荐值)
    #     'top_p': 0.95,  # top-p采样 (qwen3-vl 报告推荐值)
    #     'top_k': 20,  # top-k采样 (qwen3-vl 报告推荐值)
    #     'repetition_penalty': 1.0,  # 重复惩罚系数
    #     'presence_penalty': 0.0,  # 新内容惩罚系数
    #     'n': 1,  # 每个请求产生的回复数量
    #     'seed': 42
    # },
    
    judge_worker_num=4,
    judge_strategy=JudgeStrategy.LLM_RECALL,
    judge_model_args={
        'model_id': 'Qwen3-8B',
        'api_url': 'http://10.24.9.1:8091/v1',
        'api_key': 'EMPTY',
        'generation_config': {
            'temperature': 0.7,
            'top_p': 0.8,
            'seed': 42
        },
    },
    seed=42,
    # limit=10,        # 设置为100条数据进行测试
    stream=True,       # 是否使用流式请求，推荐设置为True防止请求超时
    timeout=6000000,
    # use_cache='output/20251220_011151',
    work_dir='./output'

)

run_task(task_cfg=task_cfg)


'''
sudo docker run -it \
      -e VACC_VISIBLE_DEVICES=0,1,2,3 \
      --privileged=true --shm-size=256g \
      --name vllm_service \
      -v /FS03/weights:/weights \
      --net=host \
      --ipc=host \
      harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-25.12.SP1 bash

vllm serve /weights/Qwen3-VL-30B-A3B-Instruct-FP8/ \
--trust-remote-code \
--tensor-parallel-size 4 \
--max-model-len 131072 \
--enforce-eager \
--port 8013 \
--served-model-name Qwen3-VL-30B-A3B-Instruct-FP8

vllm serve /weights/Qwen3-VL-30B-A3B-Thinking-FP8/ \
--trust-remote-code \
--tensor-parallel-size 4 \
--max-model-len 131072 \
--enforce-eager \
--port 8020 \
--served-model-name Qwen3-VL-30B-A3B-Thinking-FP8 \
--reasoning-parser deepseek_r1
'''