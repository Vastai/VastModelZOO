# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import os
import json
import sys
sys.path.append('.')

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, BertTokenizerFast, XLMRobertaTokenizerFast
from load_data import People_daily, compute_metrics
from xlm_roberta_embeddings import modify_xlm_roberta_embeddings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():

    parse = argparse.ArgumentParser(description="FINETUNE MODEL")
    parse.add_argument(
        "--data_name",
        type=str,
        default="people_daily",
        help="the name of the task to execute.",
    )
    parse.add_argument(
        "--model_name_or_path", 
        type=str,
        default="./code/nlp/transformers/weights/bhavikardeshna-xlm-roberta-base-chinese",
        help="pretrain model path"
    )
    parse.add_argument(
        "--num_labels", 
        type=int, 
        default=7, 
        help="class num"
    )
    parse.add_argument(
        "--seq_length", 
        type=int, 
        default=256, 
        help="max sequence length "
    )
    parse.add_argument(
        "--output_dir",
        type=str,
        default="./code/nlp/transformers/runs/Pople_for_ner_xmlroberta_base_zh",
        help="finetune model result dir.",
    )
    parse.add_argument(
        "--do_train",
        default=True,
        action='store',
        help="whether train model.",
    )
    parse.add_argument(
        "--do_eval",
        action='store',
        default=True,
        help="whether eval model.",
    )
    # parse.add_argument(
    #     "--do_predict",
    #     action='store',
    #     help="whether predict.",
    # )
    parse.add_argument(
        "--data_dir",
        type=str,
        default="./code/nlp/transformers/datasets/china-people-daily-ner-corpus",
        help="china people daily data dir",
    )
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = get_args()
    train_path = args.data_dir + '/example.train'
    dev_path = args.data_dir + '/example.dev'
    test_path = args.data_dir + '/example.test'

    output_dir = args.output_dir
    model_name_or_path = args.model_name_or_path
    num_labels = args.num_labels
    seq_length = args.seq_length
    
    tokenizer = BertTokenizerFast.from_pretrained('./code/nlp/transformers/weights/bert-base-chinese', do_lower_case=True) # create tokenizer
    # tokenizer = AutoTokenizer.from_pretrained('./code/nlp/transformers/weights/ernie3.0-xbase-zh', do_lower_case=True) # create tokenizer
    
    
    train_dataset = People_daily(train_path, tokenizer)
    dev_dataset = People_daily(dev_path, tokenizer)
    test_dataset = People_daily(test_path, tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    model = modify_xlm_roberta_embeddings(model, seq_length)
    
    training_args = TrainingArguments(
                                    output_dir=output_dir,          # output directory
                                    num_train_epochs=10,              # total number of training epochs
                                    per_device_train_batch_size=16,  # batch size per device during training
                                    per_device_eval_batch_size=64,   # batch size for evaluation
                                    warmup_steps=500,                # number of warmup steps for learning rate scheduler
                                    weight_decay=0.01,               # strength of weight decay
                                    logging_dir=output_dir + '/logs',            # directory for storing logs
                                    logging_steps=10,
                                    evaluation_strategy = "epoch",
                                    save_strategy = "epoch",
                                    load_best_model_at_end=True,
                                    metric_for_best_model="accuracy",
                                    fp16=True,
                                    )

    trainer = Trainer(
                    model=model,                         # the instantiated   Transformers model to be trained
                    args=training_args,                  # training arguments, defined above
                    train_dataset=train_dataset,         # training dataset
                    eval_dataset=test_dataset,             # evaluation dataset
                    compute_metrics=compute_metrics
                    )
    # train
    if args.do_train:
        trainer.train()
     
     # eval
    if args.do_eval:
        precision = trainer.evaluate()
        with open(os.path.join(output_dir,'eval.json'), 'w') as json_file:
            json_file.write(json.dumps(precision, ensure_ascii=False, indent=4))
        print(precision)
    