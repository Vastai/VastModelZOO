# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from modelscope.msdatasets import MsDataset

from modelscope import AutoTokenizer
import numpy as np
import copy
import os

dataset_id = "jd"
model_id = "damo/nlp_structbert_sentiment-classification_chinese-base"
# load dataset
eval_dataset = MsDataset.load(dataset_id, namespace='DAMO_NLP', split='validation').to_hf_dataset()
eval_dataset = eval_dataset.filter(lambda x: x["label"] != None and x["sentence"] != None)


tokenizer = AutoTokenizer.from_pretrained(model_id)
# NOTE: npz data format ==> (input_ids, attention_mask, token_type_ids, attention_mask, attention_mask, attention_mask)
npz_save_path = "npz_files"
calib_save_path = "calib_npz"
if not os.path.exists(npz_save_path):
    os.makedirs(npz_save_path)
if not os.path.exists(calib_save_path):
    os.makedirs(calib_save_path)


with open(f"{dataset_id}_label.txt", "w") as fw:

    for idx, data in enumerate(eval_dataset):
        inputs = tokenizer(data["sentence"], padding='max_length', max_length=256, return_tensors='pt')
        calib_data = {
            "input_ids": inputs["input_ids"].numpy().astype(np.int32),
            "attention_mask": inputs["attention_mask"].numpy().astype(np.int32),
            "token_type_ids": inputs["token_type_ids"].numpy().astype(np.int32),
        }
        save_data = copy.deepcopy(calib_data)
        save_data["attention_mask1"] = inputs["attention_mask"].numpy().astype(np.int32)
        save_data["attention_mask2"] = inputs["attention_mask"].numpy().astype(np.int32)
        save_data["attention_mask3"] = inputs["attention_mask"].numpy().astype(np.int32)

        # # gen npz data
        np.savez(os.path.join(npz_save_path, f"test_{idx}.npz"), **save_data)
        # gen label
        fw.write(f"test_{idx}.npz {int(data['label'])}\n")
        # gen calib data
        np.savez(os.path.join(calib_save_path, f"test_{idx}.npz"), **calib_data)

