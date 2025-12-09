# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from evalscope import TaskConfig
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

# 参数说明：https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/vlmevalkit_backend.html

task_cfg_dict = TaskConfig(
    work_dir='outputs',
    eval_backend='VLMEvalKit',
    eval_type=EvalType.SERVICE,
    eval_config={
        'data': ['MMBench_DEV_EN'],
        'limit': 2,
        'mode': 'all',
        'model': [
            {
                'type': 'qwen2_vl_7b_llm_28layer_2048_4096_tp4_int8',
                'api_base': 'http://0.0.0.0:9900/v1/chat/completions',
                'key': 'EMPTY',
                'name': 'CustomAPIModel',
                'temperature': 0.6,
                'img_size': -1,
                'video_llm': False,
                'max_tokens': 2048,}
            ],
        'reuse': False,
        'nproc': 1,
        'judge': 'exact_matching'}
)

run_task(task_cfg=task_cfg_dict)
