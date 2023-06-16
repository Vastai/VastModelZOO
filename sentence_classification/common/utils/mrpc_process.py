# ==============================================================================
# 
# Copyright (C) 2023 VastaiTech Technologies Inc.  All rights reserved.
# 
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :     vastml
@Time  : 2023/06/16 10:31:21
'''

import argparse
import os

import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer


def tokenizer_preprocess(examples, seq_length=128):
    tokenized_examples = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=seq_length, padding='max_length', return_tensors='pt')
    tokenized_examples['label'] = examples['label']
    return tokenized_examples


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="FINETUNE MODEL")
    parse.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="model name or path",
    )
    parse.add_argument(
        "--dataset_cache_dir", 
        type=str,
        default="../mrpc",
        help="pretrain model path"
    )
    parse.add_argument(
        "--seq_length", 
        type=int,
        default=128,
        help="sequence length"
    )
    parse.add_argument(
        "--convert_type", 
        type=str,
        default="tf",
        choices=["tf", 'hf', 'eval'],
        help="datasets convert to google or huggingface type,  choice: ['tf', 'hf, 'eval']."
    )
    parse.add_argument(
        "--save_dir", 
        type=str,
        default="./output",
        help="output to *.npz format."
    )
    args = parse.parse_args()
    raw_datasets = load_dataset("glue", "mrpc", cache_dir=args.dataset_cache_dir)
    tokenizer = BertTokenizer.from_pretrained('/home/jies/code/github_modelzoo/bert-base-uncased',  max_length=args.seq_length, padding='max_length', return_tensors='pt')

    # build the data set through the batch process method
    datasets = raw_datasets.map(tokenizer_preprocess, batched=True)

    val = datasets['validation']
    for i, data in enumerate(val):
        features = {}
        if args.convert_type == 'tf':
            features['input_ids_1'] = data['input_ids']
            features['input_mask_1'] = data['attention_mask']
            features['segment_ids_2'] = data['token_type_ids']
        elif args.convert_type == 'hf':
            features['input_ids'] = data['input_ids']
            features['attention_mask'] = data['attention_mask']
            features['token_type_ids'] = data['token_type_ids']
        elif args.convert_type == 'eval':
            features['input_0'] = data['input_ids']
            features['input_1'] = data['attention_mask']
            features['input_2'] = data['token_type_ids']
        np.savez(os.path.join(args.save_dir, str(i).zfill(5)), **features)
    
    if args.convert_type == 'eval':
        with open('npz_list.txt', 'w') as fw:
            for i in range(len(val)):
                fw.write(os.path.join(os.path.abspath(args.save_dir), str(i).zfill(5) + '.npz') + '\n')

    print('Done.')