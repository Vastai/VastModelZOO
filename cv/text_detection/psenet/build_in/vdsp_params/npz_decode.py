# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
import torch

import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../..')
from source_code.post_process import ResultFormat, Visualizer, get_result


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--gt_dir", type=str, default="/path/to/dataset/ocr/icdar2015/Challenge4/ch4_test_images", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="/path/to/code/vamc/vamp/0.2.0/outputs/dbnet", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[736, 1280], help="vamp input shape, hw")
    parse.add_argument("--save_dir", type=str, default="output/", help="")

    args = parse.parse_args()
    print(args)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()

    if 1:
        bbox_type = 'rect'
        rf = ResultFormat('PSENET_IC15', os.path.join(args.save_dir,'submit_ic15.zip'))
    else:
        bbox_type = 'poly'
        rf = ResultFormat('PSENET_CTW', os.path.join(args.save_dir,'submit_ctw'))
    
    result_map = {}
    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = os.path.join(args.gt_dir, npz_name.replace(".npz", ".jpg"))
            

            img_meta = {}
            img = cv2.imread(file_name)
            img_meta['org_img_size'] = np.array(img.shape[:2])
            img_meta['img_path'] = file_name
            img_meta['img_name'] = file_name.split('/')[-1].split('.')[0]
            img_meta['img_size'] = np.array([736, 1280])

            # load npy
            npz_file = output_npz_list[i]
            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)
            get_result(torch.tensor(heatmap), img_meta, rf, Visualizer(vis_path=os.path.join(args.save_dir,'vis')), bbox_type=bbox_type)

'''
int8
Calculated!{"precision": 0.8438520130576714, "recall": 0.7467501203659124, "hmean": 0.7923371647509578, "AP": 0}
'''