# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import evaluate


metric = evaluate.load("seqeval")

labels_list = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']
id2label = {i: k for i, k in enumerate(labels_list)}
label2id = {k: i for i, k in enumerate(labels_list)}


class People_daily(Dataset):
    def __init__(self, file_path, tokenizer, seq_length=256) -> None:
        self.file_path = file_path
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
        self.texts, self.tags = self._read_data()
        self.encodings_texts = self._get_encodings_text()
        self.encodings_labels = self._get_encodings_label()
        
        self.encodings_texts.pop("offset_mapping") # we don't want to pass this to the model

    def _read_data(self):
        file_path = Path(self.file_path)
        raw_text = file_path.read_text().strip()
        raw_docs = re.split(r'\n\t?\n', raw_text)
        #raw_docs = file_path.read_text().strip()
        token_docs = []
        tag_docs = []
        for doc in raw_docs:
            tokens = []
            tags = []
            for line in doc.split('\n'):
                token, tag = line.split(' ')
                tokens.append(token)
                tags.append(tag)
            token_docs.append(tokens)
            tag_docs.append(tags)
        
        return token_docs, tag_docs

    def _get_encodings_label(self):
        labels = [[label2id[tag] for tag in doc] for doc in self.tags]
        #print(labels)
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, self.encodings_texts.offset_mapping):
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
            arr_offset = np.array(doc_offset)
            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels

    def _get_encodings_text(self):
        return  self.tokenizer(
                            self.texts, 
                            is_split_into_words=True, 
                            return_offsets_mapping=True, 
                            padding=True, 
                            return_tensors='pt',
                            truncation=True
                            )
    
    def __getitem__(self, idx):
        item = {key: val[idx, :self.seq_length] for key, val in self.encodings_texts.items()}
        item['labels'] = torch.tensor(self.encodings_labels[idx][:self.seq_length])
        return item

    def __len__(self):
        return len(self.encodings_labels)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [labels_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [labels_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
