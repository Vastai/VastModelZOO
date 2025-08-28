# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from evalscope.constants import EvalType
from evalscope import TaskConfig, run_task

# 参数说明：https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html

task_cfg = TaskConfig(
    model='Qwen2.5-7B-Instruct-int8-tp8-2048-4096',   # 模型名称 (需要与部署时的模型名称一致)
    api_url='http://0.0.0.0:9900/v1',  # 推理服务地址
    api_key='EMPTY',
    eval_type=EvalType.SERVICE,   # 评测类型，SERVICE表示评测推理服务
    datasets=[
        'mmlu_pro','ifeval', 
        'gpqa', 'aime25'
        'live_code_bench'
    ],
    dataset_args={
        'gpqa': {
            'subset_list': [
                    'gpqa_diamond',
            ]
        }
    },
    eval_batch_size=16,       # 发送请求的并发数
    generation_config={       # 模型推理配置
        'max_tokens': 16384,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.6,   # 采样温度 (deepseek 报告推荐值)
        'top_p': 0.95,        # top-p采样 (deepseek 报告推荐值)
        'n': 1                # 每个请求产生的回复数量 (注意 lmdeploy 目前只支持 n=1)
    },
    stream=True,               # 是否使用流式请求，推荐设置为True防止请求超时
    timeout=6000000,           # 请求超时时间
    limit=0.1,               # 每个数据集最大评测数据量，不填写则默认为全部评测，可用于快速验证。支持int和float类型，int表示评测数据集的前N条数据，float表示评测数据集的前N%条数据。
    # use_cache='outputs/20250605_143533'
)

run_task(task_cfg=task_cfg)

'''
vastgenx serve --model ai300/Qwen2.5-7B-Instruct-int8-tp8-2048-4096 \
--port 9900 \
--llm_devices "[0,1,2,3,4,5,6,7]"

python VastModelZOO/docs/vastgenx/evalscope/precision_llm.py
'''