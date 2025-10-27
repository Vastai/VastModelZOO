# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../..')
from source_code.util import (calc_nme, compute_fr_and_auc, get_label, concat_output, get_config)

if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="vamp/outputs", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[256, 256], help="vamp input shape h,w")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    config = get_config("source_code/base_config.yaml")
    IONs = None
    file_list = glob.glob(os.path.join(config.data_dir, "test")+"/*.jpg")
    assert len(file_list)>0, f"FileNotFoundError: {config.data_dir}" 
    label_dict = get_label(os.path.join(config.data_dir, "test.txt"), ret_dict=True)

    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):

            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".png")
            gt_landmarks = label_dict[file_name]
            npz_file = os.path.join(args.out_npz_dir, npz_name)

            pred0 = np.load(npz_file, allow_pickle=True)["output_0"].astype("float32")
            pred1 = np.load(npz_file, allow_pickle=True)["output_1"].astype("float32")
            pred_heatmap, pred_offset = concat_output((pred0, pred1))
            sum_ion, ion = calc_nme(config, gt_landmarks, pred_heatmap, pred_offset)
            IONs = np.concatenate((IONs,ion),0) if IONs is not None else ion
        compute_fr_and_auc(IONs)  
