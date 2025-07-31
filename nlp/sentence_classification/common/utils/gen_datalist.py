# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
from typing import Dict
import argparse

import numpy as np

def gen_data_list(data_dir, save_path):
    fw = open (save_path, 'w')
    files = os.listdir(data_dir)
    files.sort()
    for f in files:
        fw.write(os.path.join(data_dir, f) + '\n')

    fw.close()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="DATA LIST GENERATE")
    parse.add_argument(
        "--data_dir",
        type=str,
        default="./code/customer_support/feiteng/bert_base_imdb/run/test_128_6inputs",
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