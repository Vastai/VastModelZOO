from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from typing import Dict
import numpy as np

from nlp_bert import Bert


class BertMRPCPairSentenceClassification(Bert):
    def __init__(self, 
        model_info: str,
        data_dir: str,
        bytes_size: int = 512,
        device_id: int = 0,
        batch_size: int = 1,
        task_name: str="MRPC",

        **kwargs
        ):
        super(BertMRPCPairSentenceClassification, self).__init__(model_info, bytes_size, device_id, batch_size)
        self.task_name = task_name
        self.data_dir = data_dir

    def get_datasets(self):
        files_len = len(os.listdir(self.data_dir))
        if files_len == 0:
            raise ValueError('dataset files is None.')
        file_name = 'test_'
        
        def dataset_loader():
            for index in range(files_len):
                npz_data = np.load(os.path.join(self.data_dir, file_name + str(index) + '.npz'))
                input_ids_1 = np.array(npz_data['input_ids_1'], dtype=np.int32)
                segment_ids_2 = np.array(npz_data['segment_ids_2'], dtype=np.int32)
                input_mask_1 = np.array(npz_data['input_mask_1'], dtype=np.int32)

                yield [input_ids_1, segment_ids_2, input_mask_1]
                
        return dataset_loader
    
    def evaluate(
        self, 
        predicts: Dict, 
        eval_path: str, 
        output_dir: str = 'output', 
        **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        fr = open(eval_path, 'r')
        contents = fr.readlines()[1:]
        fr.close()
        
        label_dict = {}
        predict_results = ''
        for index, line in enumerate(contents):
            label_dict[index] = int(line.strip().split('\t')[0])
        
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

        tpfn = [0., 0., 0., 0.]
        for key, value in predicts.items():
            predict = value[1].reshape(-1)
            if predict[0] < predict[1]:
                result = 1
            else:
                result = 0
            predict_results += str(key) + ' Reference: ' + str(label_dict[key]) + ' Predicted: ' + str(result)
            get_tpfn(result, label_dict[key], tpfn)
            predict_results += ' results_0: ' + str(predict[0]) + ' results_1: ' + str(predict[1]) + ' right: '\
                 + str(int(tpfn[0] + tpfn[3])) + ' wrong: ' + str(int(tpfn[1] + tpfn[2])) + '\n'

        f1, acc = get_f1(tpfn)
        self.f1 = f1
        self.acc = acc

        print("\nF1: ", round(f1, 4), "Accuracy:", round(acc, 4))
        predict_results += "F1: " + str(round(f1, 5)) + " Accuracy: " + str(round(acc, 5))
        fw = open(os.path.join(output_dir, f'{self.task_name}_evaluate.txt'), 'w')
        fw.write(predict_results)
        fw.close()



if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="RUN NLP-BERT-CLS WITH VACL")
    parse.add_argument(
        "--task_name",
        type=str,
        default="mrpc",
        help="the name of the task to execute.",
    )
    parse.add_argument(
        "--model_info", 
        type=str,
        default="../../../nlp/bert/model_info/model_info_bert.json",
        help="model info: weight of model quantization and pipline file."
    )
    parse.add_argument(
        "--eval_path", 
        type=str, 
        default="../../../data/test/MRPC/dev.tsv", 
        help="validation file."
    )
    parse.add_argument("--device_id", type=int, default=0, help="device id")
    parse.add_argument(
        "--bytes_size", 
        type=int, 
        default=512, 
        help="max sequence is converted to the length of bytes."
    )
    parse.add_argument(
        "c",
        type=str,
        default="../../../data/test/MRPC/dev408",
        help="the data directory of the type of bytes to be predicted, the data type is .npz",
    )
    parse.add_argument("--batch_size", type=int, default=1, help="bacth size")
    parse.add_argument("--save_dir", type=str, default="output", help="save the result path")
    args = parse.parse_args()

    # create bert classifer model
    bert_cls = BertMRPCPairSentenceClassification(
        model_info=args.model_info,
        data_dir = args.data_dir,
        bytes_size=args.bytes_size,
        device_id=args.device_id,
        batch_size=args.batch_size,
        task_name=args.task_name
    )
    
    # build datasets iterator
    datasets = bert_cls.get_datasets()

    # batch run model
    results = bert_cls.run_batch(datasets())
    predicts = {}
    for i, result in enumerate(results):
        predicts[i] = result
        print(predicts[i])
        
    # evaluate and output results
    bert_cls.evaluate(predicts, args.eval_path, args.save_dir)
    