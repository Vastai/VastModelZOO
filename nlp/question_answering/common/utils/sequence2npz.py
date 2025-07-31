# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import numpy as np
import argparse

parse = argparse.ArgumentParser(description="DATA FORMAT CONVERT")
parse.add_argument(
    "--data_dir",
    type=str,
    default="./datasets/SQuAD_1.1/val_npz/",
    help="MRPC-dev *.bin data path",
)
parse.add_argument(
    "--save_dir", 
    type=str,
    default="./modelzoo/vastmodelzoo/question_answering/datasets/squad_1,1/",
    help="output *.npz and npz_datalist.txt path"
)
args = parse.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
length = len(os.listdir(args.data_dir))
fw = open(os.path.join(os.path.dirname(args.save_dir), 'npz_datalist.txt'), 'w')
for i in range(length):
    name = 'test_' + str(i) + '.npz'
    npz_path = os.path.join(args.data_dir, name)
    npz_data = np.load(npz_path)
    features = {}
    
    features['input_0'] = npz_data[npz_data.files[0]].reshape(1, -1)
    features['input_1'] = npz_data[npz_data.files[2]].reshape(1, -1)
    features['input_2'] = npz_data[npz_data.files[1]].reshape(1, -1)
    
    save_path = os.path.join(args.save_dir, name)
    np.savez(save_path, **features)
    fw.write(save_path + '\n')
    print(save_path)
    
fw.close()