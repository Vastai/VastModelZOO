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
import json
import os
from packaging import version

import torch
import torch.nn as nn
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer


def tokenizer_preprocess(examples):
    tokenized_examples = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=seq_length, padding='max_length', return_tensors='pt')
    tokenized_examples['label'] = examples['label']
    return tokenized_examples


def get_tpfn(pre:int, label:int, tpfn: list):
    if pre == 1 and label == 1:
        tpfn[0] += 1
    elif pre == 1 and label == 0:
        tpfn[1] += 1
    elif pre == 0 and label == 1:
        tpfn[2] += 1
    else:
        tpfn[3] += 1
        
        
def get_f1(tpfn: list):
    precision = tpfn[0] / (tpfn[0] + tpfn[1] + 1E-6)
    recall = tpfn[0] / (tpfn[0] + tpfn[2] + 1E-6)
    f1 = 2. * precision * recall / (precision + recall + 1E-6)
    acc = ( tpfn[0] +  tpfn[3]) / ( tpfn[0] +  tpfn[1] +  tpfn[2] +  tpfn[3])
    return f1, acc
  
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    tpfn = [0., 0., 0., 0.]
    for pre, lbl in zip(predictions, labels):
      get_tpfn(pre, lbl, tpfn)
    
    f1, acc = get_f1(tpfn)
    # score = accuracy_metric.compute(predictions=predictions, references=labels)
    # return score
    return {
              "eval_f1": f1,
              "eval_accuracy": acc
           }


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

        self.seq_length = config.max_seq_length

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        # seq_length = input_shape[1]
        seq_length = self.seq_length
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    
def modify_bert_embeddings(model, max_seq=128):
    model.config.max_seq_length = max_seq
    embeddings_new = BertEmbeddings(model.config)
    embeddings_new.word_embeddings.weight = model.bert.embeddings.word_embeddings.weight
    embeddings_new.position_embeddings.weight = model.bert.embeddings.position_embeddings.weight
    embeddings_new.token_type_embeddings.weight = model.bert.embeddings.token_type_embeddings.weight
    embeddings_new.LayerNorm.weight = model.bert.embeddings.LayerNorm.weight
    embeddings_new.dropout = model.bert.embeddings.dropout
    embeddings_new.position_embedding_type = model.bert.embeddings.position_embedding_type
    model.bert.embeddings = embeddings_new
    return model


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="FINETUNE MODEL")
    parse.add_argument(
        "--data_name",
        type=str,
        default="mrpc",
        help="the name of the task to execute.",
    )
    parse.add_argument(
        "--model_name_or_path", 
        type=str,
        default="/home/jies/code/nlp/transformers/runs/roberta-base/checkpoint-2300",
        help="pretrain model path"
    )
    parse.add_argument(
        "--num_labels", 
        type=int, 
        default=2, 
        help="class num"
    )
    parse.add_argument(
        "--seq_length", 
        type=int, 
        default=128, 
        help="max sequence length "
    )
    parse.add_argument(
        "--output_dir",
        type=str,
        default="/home/jies/code/nlp/transformers/runs/bert_test",
        help="finetune model result dir.",
    )
    parse.add_argument(
        "--do_train",
        action='store',
        help="whether train model.",
    )
    parse.add_argument(
        "--do_eval",
        action='store',
        help="whether eval model.",
    )
    parse.add_argument(
        "--do_predict",
        action='store',
        help="whether predict.",
    )
    args = parse.parse_args()
    
    output_dir = args.output_dir
    model_name_or_path = args.model_name_or_path  # model_name 
    num_labels = args.num_labels
    seq_length = args.seq_length

    # load datasets
    datasets = load_dataset('glue', args.data_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # build the data set through the batch process method
    datasets = datasets.map(tokenizer_preprocess, batched=True)

    # build the accuracy evaluation method
    accuracy_metric = evaluate.load("accuracy")

    # load seq cls pretrain model
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    model = modify_bert_embeddings(model, seq_length)
    
    # create train args
    args = TrainingArguments(
                            learning_rate=2e-5,
                            per_device_train_batch_size=8,
                            per_device_eval_batch_size=128,
                            num_train_epochs=20,
                            weight_decay=0.01,
                            output_dir=output_dir,
                            logging_steps=10,
                            evaluation_strategy = "epoch",
                            save_strategy = "epoch",
                            load_best_model_at_end=True,
                            metric_for_best_model="accuracy",
                            fp16=True,
                            )
    # create trainner
    trainer = Trainer(
                    model,
                    args=args,
                    train_dataset=datasets["train"],
                    eval_dataset=datasets["validation"],
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
                    )

    # train
    if args.do_train:
      out = trainer.train()
      
    # eval
    if args.do_eval:
      precision = trainer.evaluate()
      with open(os.path.join(output_dir,'eval.json'), 'w') as json_file:
          json_file.write(json.dumps(precision, ensure_ascii=False, indent=4))

    if args.do_predict:
      # predict and write results
      result = trainer.predict(datasets["test"])
      predictions = result.predictions
      predictions = predictions.argmax(axis=-1)
      label_ids = result.label_ids
      
      with open(os.path.join(output_dir, 'predict.txt'), 'w') as fw:
        for pre, lab in zip(predictions, label_ids):
          fw.write(str(pre) + ' ' + str(lab) + '\n')