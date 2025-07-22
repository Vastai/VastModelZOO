import os
from typing import Dict
import argparse

import numpy as np
from seqeval.metrics import accuracy_score, f1_score


class BertChineseNER:
    def __init__(
        self, 
        pre_dir,
        eval_path,
        task_name: str="NER",
        ):
        
        self.eval_path = eval_path
        self.task_name = task_name
        self.pre_dir = pre_dir
        self.categories = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']
        
        self.pre_dict = self._get_predict()

    def _read_eval_file(self):
        data_list= []
        assert os.path.isfile(self.eval_path), "The eval file does not exist. Please check whether the file path is correct."
        with open(self.eval_path) as fr:
            for index, line in enumerate(fr):
                line0 = ((line.strip('\n')).lstrip('[')).rstrip(']')
                line1 = line0.split(',')
                temp = []
                for i in range(len(line1)):
                    temp.append(int(line1[i]))
                data_list.append(np.array(temp))
        return data_list
    
    def _get_predict(self):
        file_names = os.listdir(self.pre_dir)
        file_names.sort()
        predict_dict = {}
        for i, name in enumerate(file_names):
            path = os.path.join(self.pre_dir, name)
            npz_data = np.load(path)
            predict = npz_data['output_0']
            predict_dict[i] = predict
        
        return predict_dict

    def evaluate(self):
        eval_list = self._read_eval_file() 
        predicts = self.pre_dict
        # The feature map is converted to the same sequence length as the validation label
        # , and the maximum index for each sequence position is obtained.
        predict_list = []
        for k, v in predicts.items():
            predict_list.append(np.argmax(v.reshape(256, 7), axis=1))

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
        print('F1: ' , f1)
        print('accuracy: ', accuracy)
        

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="MPRC EVAL")
    parse.add_argument(
        "--result_dir",
        type=str,
        default="./result/ner",
        help="vamp output *.npz results path",
    )
    parse.add_argument(
        "--label_path", 
        type=str,
        default="./instance_Peoples_Daily.txt",
        help="china-people-daily-ner-corpus ground-true path"
    )
    args = parse.parse_args()
    bert_eval = BertChineseNER(args.result_dir, args.label_path)
    bert_eval.evaluate()