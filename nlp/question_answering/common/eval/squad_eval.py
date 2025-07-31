# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from typing import Dict
import json
import sys
import os
import argparse

import numpy as np
from metrics import prediction_score, evaluate

def eval_squad(
        predicts: Dict, 
        eval_path: str, 
        vocab_path: str, 
        debug_sample: int = 10570, 
        output_dir: str = 'output',
        **kwargs
        ) -> None:
        """Based on the validation dataset, the prediction results are evaluated"""
        expected_version = '1.1'
        unique_id = 1000000000
        result_list = []

        for key, value in predicts.items():
            # print("value shape:", value.shape)
            output = np.array(value[:, :2], dtype=np.float32).reshape(384, 2)
                
            output1 = output[:, 0]
            output2 = output[:, 1]

            start_index = np.argmax(output1)
            end_index = np.argmax(output2)
            
            print("index {}, pred: start {}, end {}".format(key + unique_id,  start_index, end_index))
            result_dict = {}
            result_dict["unique_ids"] = key + unique_id
            result_dict["start_logits"] = output1
            result_dict["end_logits"] = output2
            result_list.append(result_dict)

        prediction_score(result_list, eval_path, debug_sample, vocab_path, output_dir)
        
        with open(eval_path) as dataset_file:
            dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
        with open(os.path.join(output_dir, "predictions.json")) as prediction_file:
            predictions = json.load(prediction_file)
        print(json.dumps(evaluate(dataset, predictions)))
        

def get_predcit(pre_dir):
    file_names = os.listdir(pre_dir)
    file_names.sort()
    predict_dict = {}
    for i, name in enumerate(file_names):
        path = os.path.join(pre_dir, name)
        npz_data = np.load(path)
        predict = npz_data['output_0']
        predict_dict[i] = predict
    
    return predict_dict


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="SQUAD EVAL")
    parse.add_argument(
        "--result_dir",
        type=str,
        default="./out_result/bert_squad_output",
        help="vamp output *.npz results path",
    )
    parse.add_argument(
        "--eval_path", 
        type=str,
        default="./dataset/squad/dev-v1.1.json",
        help="MRPC-dev file path "
    )
    parse.add_argument(
        "--vocab_path", 
        type=str,
        default="./model/bert_base_uncased/vocab.txt",
        help="MRPC-dev file path "
    )
    args = parse.parse_args()
    
    eval_squad(
        get_predcit(args.result_dir),
        eval_path=args.eval_path,
        vocab_path=args.vocab_path
        )