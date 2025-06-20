# ==============================================================================
#
# Copyright (C) 2024 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :          lance
@Email : lance.wang@vastaitech.com
@Time  : 	2025/04/21 19:43:31
'''

import datasets

## https://huggingface.co/datasets/mteb/sts12-sts/tree/main
## en dataset

dataset = datasets.load_dataset("mteb/sts12-sts", split="test")
print(dataset)
export_path = 'mteb-sts12-sts_test.jsonl'
dataset.to_json(export_path, force_ascii=False, num_proc=1)

## https://huggingface.co/datasets/C-MTEB/BQ/tree/main
## zh dataset

dataset = datasets.load_dataset("C-MTEB/BQ", split="test")
print(dataset)
export_path = 'c-mteb-bq_test.jsonl'
dataset.to_json(export_path, force_ascii=False, num_proc=1)


## https://huggingface.co/datasets/zyznull/msmarco-passage-ranking

dataset = datasets.load_dataset("zyznull/msmarco-passage-ranking", split="dev")
print(dataset)
export_path = 'zyznull-msmarco-passage-ranking_dev.jsonl'
dataset.to_json(export_path, force_ascii=False, num_proc=1)

