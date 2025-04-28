
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :    zpwen
@Email  :    algorithm@vastaitech.com
@Time   :    2025/04/24 16:17:38
'''

import os
import math
import cv2
import copy
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.nn import functional as F


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="datasets/PPM-100/image", help="hr gt image folder")
    parse.add_argument("--gt_dir", type=str, default="datasets/PPM-100/matte", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="datasets/PPM-100/npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="./outputs", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[480, 288], help="vamp input shape, h,w")
    parse.add_argument("--draw_dir", type=str, default="./npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=False, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()

    # metrics
    metrics_ins = {}
    metrics_data = {}
    metrics = ['mse', 'mad', 'grad', 'conn']

    # add seg_metrics
    import sys
    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + '../../source_code/') # 指向源仓库路径
    from metrics import metrics_class_dict, StreamSegMetrics
    seg_metrics = StreamSegMetrics(2)

    for key in metrics:
        key = key.lower()
        metrics_ins[key] = metrics_class_dict[key]()
        metrics_data[key] = None

    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".jpg")

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, os.path.basename(npz_name))

            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)
            
            pred = (np.squeeze(heatmap[0]) * 255).astype('uint8')
            # eval
            gt = cv2.imread(os.path.join(args.gt_dir, os.path.basename(file_name)), 0)
            gt = cv2.resize(gt, (args.input_shape[1], args.input_shape[0]), interpolation=cv2.INTER_AREA)
            save_path = os.path.join(args.draw_dir, os.path.basename(file_name))
            con = np.concatenate([gt, pred], axis=1)
            cv2.imwrite(save_path, con)

            for key in metrics_ins.keys():
                metrics_data[key] = metrics_ins[key].update(pred, gt, trimap=None)
            
            # add seg_metrics 
            gt[gt < 128] = 0
            gt[gt >= 128] = 1
            pred[pred < 128] = 0
            pred[pred >= 128] = 1
            seg_metrics.update(gt, pred)


        for key in metrics_ins.keys():
            metrics_data[key] = metrics_ins[key].evaluate()
        print("matting_metrics: \n", metrics_data)

        # add seg_metrics
        val_score = seg_metrics.get_results()
        print("seg_metrics: \n", seg_metrics.to_str(val_score))

"""
modnet-fp16-none-3_480_288-vacc
matting_metrics: 
 {'mse': 0.03408449085087465, 'mad': 0.039186252099219304, 'grad': 6.205589920349121, 'conn': 5.3696625666809075}
seg_metrics: 
 
Overall Acc: 0.961954
Mean Acc: 0.944652
FreqW Acc: 0.926454
Mean IoU: 0.912700

"""