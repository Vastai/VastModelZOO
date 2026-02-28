# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from evalscope import TaskConfig
from evalscope.run import run_task

task_cfg = TaskConfig(
    work_dir='./output_retrieval_reranking_env2',
    eval_backend='RAGEval',
    eval_config={
        'tool': 'MTEB',
        'model': [
            {
                'model_name': 'bge-m3',
                'api_base': 'http://192.168.28.113:8001/v1/',
                'api_key': 'EMPTY',
                "dimensions": None,
                'encode_kwargs': {
                    'batch_size': 1,
                },
                "model_kwargs": {
                    'instruction_template': 'Instruct: {}\nQuery:',
                    'instruction_dict_path': 'task_prompts.json'
                }
            }
        ],
        'eval': {
            'tasks': [
                "ArguAna",
                "MindSmallReranking"
            ],
            'output_folder': 'results_retrieval_reranking_env2',
            'verbosity': 2,
            'overwrite_results': True,
        },
    },
)


# Run task
run_task(task_cfg=task_cfg) 
