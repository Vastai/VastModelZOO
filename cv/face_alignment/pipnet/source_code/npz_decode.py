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
import glob
import argparse
import numpy as np
from tqdm import tqdm
import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../..')
from source_code.utils import (post_process, get_meanface, compute_nme, compute_fr_and_auc,
                   get_label)
import torch

if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="vamp/outputs", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[256, 256], help="vamp input shape h,w")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()

    norm_indices = [60, 72]
    num_lms = 98
    num_nb = 10
    net_stride = 32
    meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join(args.data_dir, 'meanface.txt'), num_nb)
    nmes_merge = []
    image_files = []
    
    labels = get_label(os.path.join(args.data_dir, "test.txt"), ret_dict=True)

    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):

            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".png")
            lms_gt = labels[file_name]
            norm = np.linalg.norm(lms_gt.reshape(-1, 2)[norm_indices[0]] - lms_gt.reshape(-1, 2)[norm_indices[1]])

            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            pred0 = np.load(npz_file, allow_pickle=True)["output_0"].astype("float32")
            pred1 = np.load(npz_file, allow_pickle=True)["output_1"].astype("float32")
            pred2 = np.load(npz_file, allow_pickle=True)["output_2"].astype("float32")
            pred3 = np.load(npz_file, allow_pickle=True)["output_3"].astype("float32")
            pred4 = np.load(npz_file, allow_pickle=True)["output_4"].astype("float32")
            # print(pred0.shape)
            # print(pred4.shape)
            pred = (pred0, pred1, pred2, pred3, pred4)
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = post_process(pred, num_nb, net_stride)
            
            lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
            tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(num_lms, max_len)
            tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(num_lms, max_len)
            tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
            tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
            lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()

            lms_pred = lms_pred.cpu().numpy()
            lms_pred_merge = lms_pred_merge.cpu().numpy()

            nme_merge = compute_nme(lms_pred_merge, lms_gt, norm)
            nmes_merge.append(nme_merge)
        

        print('Image num:', len(labels))
        print('nme: {}'.format(np.mean(nmes_merge)))
        fr, auc = compute_fr_and_auc(nmes_merge)
        print('fr: {}'.format(fr))
        print('auc: {}'.format(auc))
