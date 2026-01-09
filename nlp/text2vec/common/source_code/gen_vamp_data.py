# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import torch
import datasets
import numpy as np
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("modelzoo/text2vec/bge/bge-m3") # BAAI/bge-m3
sentence_batch = ['Hybrid retrieval leverages the strengths of various methods',
                  '它同时支持 embedding 和稀疏检索，这样在生成稠密 embedding 时，无需付出额外代价，就能获得与 BM25 类似的 token 权重',
                  '我们建议使用以下流程：混合检索+重新排名',
                  "测试句子好不好用，因为我做了修改你知道吗",
                  "Different from embedding model, reranker uses question and document as input and directly output similarity instead of embedding. You can get a relevance score by inputting query and passage to the reranker. And the score can be mapped to a float value in [0,1] by sigmoid function.",
                  "Multi-Functionality: It can simultaneously perform the three common retrieval functionalities of embedding model: dense retrieval, multi-vector retrieval, and sparse retrieval.",
                  'Multi-Linguality: It can support more than 100 working languages.'
                  "Multi-Granularity: It is able to process inputs of different granularities, spanning from short sentences to long documents of up to 8192 tokens.",
                  "Hybrid retrieval leverages the strengths of various methods, offering higher accuracy and stronger generalization capabilities. A classic example: using both embedding retrieval and the BM25 algorithm. Now, you can try to use BGE-M3, which supports both embedding and sparse retrieval. This allows you to obtain token weights (similar to the BM25) without any additional cost when generate dense embeddings. To use hybrid retrieval, you can refer to Vespa and Milvus.",
                  "在这个项目中，我们发布了BGE-M3，它是第一个具有多功能、多语言和多粒度特性的文本检索模型。多功能:可以同时执行三种检索功能：单向量检索、多向量检索和稀疏检索。多语言:支持100多种工作语言。多粒度:它能够处理不同粒度的输入，从短句子到长达8192个词汇的长文档。"
                  ]

# en dataset
# dataset = datasets.load_dataset('json', data_files='vacc_deploy/mteb-sts12-sts_test.jsonl', split='train')
# sentence_batch= dataset['sentence1'][:100]

# zh dataset
# dataset = datasets.load_dataset('json', data_files='vacc_deploy/c-mteb-bq_test.jsonl', split='train')
# sentence_batch= dataset['sentence1'][:100]

npz_save_path = 'input_npz'
os.makedirs(npz_save_path, exist_ok=True)

with open('vamp_npz_list.txt', 'w') as f:
    for index, input_text in enumerate(sentence_batch):
        inputs = tokenizer(
            [input_text],
            truncation=True,
            return_tensors="pt",
            padding='max_length',
            max_length=512)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = torch.zeros_like(attention_mask)

        save_data = {'input_ids': input_ids.numpy().astype(np.int32),
                    'attention_mask': attention_mask.numpy().astype(np.int32),
                    'token_type_ids': token_type_ids.numpy().astype(np.int32),
                    'attention_mask1': attention_mask.numpy().astype(np.int32),
                    'attention_mask2': attention_mask.numpy().astype(np.int32),
                    'attention_mask3': attention_mask.numpy().astype(np.int32)
                    }
        file_path = os.path.join(npz_save_path, f"{index}.npz")
        f.write(file_path + '\n')
        np.savez(file_path, **save_data)