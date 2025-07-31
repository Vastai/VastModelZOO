# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import argparse


def gen_data_list(data_dir, save_path):
    fw = open (save_path, 'w')
    files_len = len(os.listdir(data_dir))
    for i in range(files_len):
        fw.write(os.path.join(data_dir, f'test_{str(i)}.npz') + '\n')

    fw.close()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="DATA LIST GENERATE")
    parse.add_argument(
        "--data_dir",
        type=str,
        default="./code/vmc/datasets/nlp/squad_v1_1/val_npz_6inputs",
        help="data dir",
    )
    parse.add_argument(
        "--save_path", 
        type=str,
        default="./npz_datalist.txt",
        help="output *.npz and npz_datalist.txt path"
    )
    args = parse.parse_args()
    
    gen_data_list(args.data_dir, args.save_path)