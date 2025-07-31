# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import argparse
import numpy as np


def eval(p, r):
    p_index = np.argmax(p, axis=-1)[0]
    if p_index == r:
        return 1
    else:
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_path", type=str, default='/opt/telecom/datasets/data_sanyong/sst2_eval_labels.txt', help='sst2 validation-set label file')
    parser.add_argument("--result_dir", type=str, default='/opt/telecom/save/telecom_int8_max_128_sanyong/', help='runstream result save path')
    args = parser.parse_args()
    labels_path = args.labels_path
    result_dir = args.result_dir
    
    with open(labels_path, 'r') as fr:
        labels = fr.readlines()
    
    pos_num = 0.
    for i in range(len(os.listdir(result_dir))):
        output_file = os.path.join(result_dir, "output_" + str(i).zfill(6) + '.npz')
        output = np.load(output_file)
        pos_num += eval(output['output_0'],int(labels[i]))
    
    print('acc: ', pos_num / (i+1))
        

    
    