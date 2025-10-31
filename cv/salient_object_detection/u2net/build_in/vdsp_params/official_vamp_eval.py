# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

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


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="./dataset/sod/ECSSD/image", help="hr gt image folder")
    parse.add_argument("--gt_dir", type=str, default="./dataset/sod/ECSSD/mask", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="./dataset/sod/ECSSD/npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="vamp_outputs", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[320, 320], help="vamp input shape")
    parse.add_argument("--draw_dir", type=str, default="npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()

    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".jpg")

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            heatmap = np.load(output_npz_list[i], allow_pickle=True)["output_0"].astype(np.float32)
            
            heatmap = np.expand_dims(heatmap, 0)
            pred = torch.from_numpy(heatmap)
            # pred = 1.0 - pred[:,0,:,:]
            pred = normPRED(pred)
            mask = pred.squeeze()
            mask = mask.data.numpy()
            mask = Image.fromarray(mask*255).convert('RGB')
            
            save_path = os.path.join(args.draw_dir, os.path.basename(file_name))
            mask.save(save_path)

