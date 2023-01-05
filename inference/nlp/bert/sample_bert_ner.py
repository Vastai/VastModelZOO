from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from typing import Dict

import numpy as np
from seqeval.metrics import accuracy_score, f1_score

from nlp_bert import Bert


class BertChineseNER(Bert):
    def __init__(self, 
        model_info: str,
        data_dir: str,
        bytes_size: int = 512,
        device_id: int = 0,
        batch_size: int = 1,
        task_name: str="ner",
        **kwargs
        ):
        super(BertChineseNER, self).__init__(model_info, bytes_size, device_id, batch_size)
        self.task_name = task_name
        self.data_dir = data_dir
        self.categories = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']

    def get_datasets(self):
        files_len = len(os.listdir(self.data_dir))
        if files_len == 0:
            raise ValueError('dataset files is None.')
        file_name = 'test_'
        
        def dataset_loader():
            for index in range(files_len):
                npz_data = np.load(os.path.join(self.data_dir, file_name + str(index) + '.npz'))
                input_ids = np.array(npz_data['input_ids'], dtype=np.int32)
                segment_ids = np.array(npz_data['segment_ids'], dtype=np.int32)
                input_mask = np.array(npz_data['input_mask'], dtype=np.int32)
                yield [input_ids, segment_ids, input_mask]
                
        return dataset_loader
    
    def read_eval_file(self, eval_path):
        assert os.path.isfile(eval_path), "The eval file does not exist. \
            Please check whether the file path is correct."
        
        data_dict = []
        with open(eval_path) as fr:
            for index, line in enumerate(fr):
                line0 = ((line.strip('\n')).lstrip('[')).rstrip(']')
                line1 = line0.split(',')
                temp = []
                for i in range(len(line1)):
                    temp.append(int(line1[i]))
                data_dict.append(np.array(temp))
                
        return data_dict

    def evaluate(
        self, 
        predicts: Dict, 
        eval_path: str,
        output_dir: str = 'output', 
        **kwargs):

        os.makedirs(output_dir, exist_ok=True)
        eval_list = self.read_eval_file(eval_path) 

        # The feature map is converted to the same sequence length as the validation label
        # , and the maximum index for each sequence position is obtained.
        predict_list = []
        for k, v in predicts.items():
            predict_list.append(np.argmax(np.reshape(v[1], v[0]), axis=1))

        y_true = []
        y_pred = []

        categories_id2label = {i: k for i, k in enumerate(self.categories)}
        label_map = categories_id2label

        for index, label in enumerate(eval_list):
            temp_1 = []
            temp_2 = []
            for j in range(label.shape[0]):
                if j == 0:
                    continue
                elif eval_list[index][j] == len(label_map) - 1:
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label[j]])
                    temp_2.append(label_map[predict_list[index][j]])
        
        f1 = f1_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        print('\nF1: ' , f1)
        print('accuracy: ', accuracy)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="RUN NLP-BERT-CLS WITH VACL")
    parse.add_argument(
        "--task_name",
        type=str,
        default="china_people_daily",
        help="the name of the task to execute.",
    )
    parse.add_argument(
        "--model_info", 
        type=str,
        default="../../../nlp/bert/model_info/model_info_bert.json",
        help="weight of model quantization and pipline file."
    )
    parse.add_argument(
        "--eval_path", 
        type=str, 
        default="../../../data/test/china-people-daily-ner-corpus/example.test", 
        help="validation file."
    )
    parse.add_argument("--device_id", type=int, default=0, help="device id")
    parse.add_argument(
        "--bytes_size", 
        type=int, 
        default=1024, 
        help="max sequence is converted to the length of bytes."
    )
    parse.add_argument(
        "--data_dir",
        type=str,
        default="../../../data/test/china-people-daily-ner-corpus/test4636",
        help="the data directory of the type of bytes to be predicted, the data type is .npz",
    )
    parse.add_argument("--batch_size", type=int, default=1, help="bacth size")
    parse.add_argument("--save_dir", type=str, default="output", help="save the result path")
    args = parse.parse_args()

     # create bert ner model
    bert_ner = BertChineseNER(
        model_info=args.model_info,
        data_dir=args.data_dir,
        bytes_size=args.bytes_size,
        device_id=args.device_id,
        batch_size=args.batch_size,
        task_name=args.task_name
    )
    
    # build datasets iterator
    datasets = bert_ner.get_datasets()

    # batch run model
    results = bert_ner.run_batch(datasets())
    predicts = {}
    for i, result in enumerate(results):
        predicts[i] = result
        print(predicts[i])
        
    # evaluate and output results
    bert_ner.evaluate(predicts, args.eval_path, args.save_dir)
    