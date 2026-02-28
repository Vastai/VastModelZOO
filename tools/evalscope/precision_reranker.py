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
    work_dir='./rerank_results',
    eval_backend='RAGEval',
    eval_config={
        'tool': 'MTEB',
        'model': [
            {
                'model_name': 'bge-reranker-v2-m3',
                'api_base': 'http://192.168.28.113:8012/v1/rerank',
                'api_key': 'EMPTY',
                'encode_kwargs': {
                    'batch_size': 1,
                },
                'is_cross_encoder': True,
                "model_kwargs": {
                    'instruction_dict_path': 'task_prompts.json',
                    'task_name': ['VideoRetrieval',
                                  'EcomRetrieval',
                                  'MedicalRetrieval'
                                 ],
                    'embed_results':['/path/to/VideoRetrieval_default_predictions.json', 
                                        '/path/to/EcomRetrieval_default_predictions.json', 
                                        '/path/to/MedicalRetrieval_default_predictions.json'
                                        ]
                }
            }
        ],
        'eval': {
            'tasks': [
                    "VideoRetrieval",
                    "EcomRetrieval",
                    "MedicalRetrieval",
            ],
            'top_k': 100,
            'verbosity': 2,
            'output_folder': './rerank_results',
            'overwrite_results': True,
        },
    },
)


# Run task
run_task(task_cfg=task_cfg) 