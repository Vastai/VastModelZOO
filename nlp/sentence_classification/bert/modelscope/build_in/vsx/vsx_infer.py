# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import vaststreamx as vsx
import numpy as np
import argparse
import os

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

class ModelBase:
    def __init__(
        self, model_prefix, vdsp_config, batch_size=1, device_id=0, hw_config=""
    ) -> None:
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.preproc_ops_ = vsx.Operator.load_ops_from_json_file(vdsp_config)
        assert len(self.preproc_ops_) > 0, "load_ops_from_json_file failed"
        self.graph_ = vsx.Graph()
        self.graph_.add_operators(*self.preproc_ops_, self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

    @property
    def batch_size(self):
        return self.model_.batch_size

    @property
    def max_batch_size(self):
        return self.model_.max_batch_size

    @property
    def input_count(self):
        return self.model_.input_count

    @property
    def output_count(self):
        return self.model_.output_count

    @property
    def input_shape(self):
        return self.model_.input_shape

    @property
    def output_shape(self):
        return self.model_.output_shape

    def process_impl(self, inputs):

        vsx_tensors = [[
            vsx.from_numpy(
                np.array(input, dtype=np.int32), self.device_id_
            ) for _, input in inputs.items()
        ]]

        outputs = self.stream_.run_sync(vsx_tensors)
        return [[vsx.as_numpy(o).astype(np.float32) for o in out] for out in outputs]

def get_acc(p, l):
    metric = p==l
    return metric.sum() / len(metric)

def evaluate(preds_max, labels):
    _, _, f1, _ = precision_recall_fscore_support(labels, preds_max, average='binary')
    acc =  get_acc(labels, preds_max)

    print(f"'accuracy': {acc}, 'binary-f1': {f1}")


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="RUN NLP-BERT-CLS WITH VACL")
    parse.add_argument(
        "--model_prefix",
        type=str,
        default="bert_int32-fp16-none-1_256_1_256_1_256-vacc/mod",
        help="path to vacc params",
    )
    parse.add_argument(
        "--vdsp_json",
        type=str,
        default="vdsp.json",
        help="path to vdsp json"
    )
    parse.add_argument(
        "--data_dir",
        type=str,
        default="npz_files",
        help="path to npz data",
    )
    parse.add_argument(
        "--label",
        type=str,
        default="jd_label.txt",
        help="path to gt file",
    )
    parse.add_argument("--input_seq_len", type=int, default=256, help="device id")
    parse.add_argument("--device_id", type=int, default=0, help="device id")
    parse.add_argument("--batch_size", type=int, default=1, help="bacth size")

    args = parse.parse_args()
    # init vsx model
    model = ModelBase(
        args.model_prefix,
        args.vdsp_json,
        args.batch_size,
        args.device_id
    )
    # load gt
    gt_list = []
    with open(args.label, "r") as fr:
        for line in fr:
            name, gt = line.strip().split()
            gt_list.append(int(gt))
    # load dataset
    result_list = []
    for idx in tqdm(range(len(gt_list))):
        np_array = np.load(os.path.join(args.data_dir, f"test_{idx}.npz"))
        output = model.process_impl(np_array)[0]
        result_list.append(np.argmax(output[0]))
    evaluate(np.array(result_list), np.array(gt_list))


#######################################
# torch {'accuracy': 0.9200080192461909, 'binary-f1': 0.9185548071034906, 'f1': 0.9185548071034906}
# fp16 'accuracy': 0.9196070569366479, 'binary-f1': 0.9180796731358529

