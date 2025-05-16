
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
import cv2
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.nn import functional as F


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def convert_label(label, inverse=False):
    ignore_label = 255
    label_mapping = {-1: ignore_label, 0: ignore_label, 
                            1: ignore_label, 2: ignore_label, 
                            3: ignore_label, 4: ignore_label, 
                            5: ignore_label, 6: ignore_label, 
                            7: 0, 8: 1, 9: ignore_label, 
                            10: ignore_label, 11: 2, 12: 3, 
                            13: 4, 14: ignore_label, 15: ignore_label, 
                            16: ignore_label, 17: 5, 18: ignore_label, 
                            19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                            25: 12, 26: 13, 27: 14, 28: 15, 
                            29: ignore_label, 30: ignore_label, 
                            31: 16, 32: 17, 33: 18}
    
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="./dataset/seg/cityscapes/leftImg8bit/val", help="hr gt image folder")
    parse.add_argument("--gt_dir", type=str, default="./dataset/seg/cityscapes/gtFine/val", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="./dataset/seg/cityscapes/npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="./code/vamc/vamp/0.2.0/outputs/bisenetv2", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[736, 960], help="vamp input shape, h,w")
    parse.add_argument("--draw_dir", type=str, default="./npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()

    num_classes = 19
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    import sys
    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + '../../')
    colors = np.loadtxt("vacc_code/runmodel/cityscapes_colors.txt").astype('uint8')

    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_sub_name = line.split("/")[-2] + "/" +  line.split("/")[-1].strip('\n')
            file_name = npz_sub_name.replace(".npz", ".png")

            # src image
            ori_image  = Image.open(os.path.join(args.src_dir, file_name))

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, os.path.basename(npz_sub_name))

            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)

            # draw
            tvm_predict = torch.from_numpy(heatmap)
            tvm_predict = F.interpolate(tvm_predict, args.input_shape, mode='bilinear')     
            
            predict = tvm_predict[0].cpu().numpy()
            predict = np.asarray(predict, dtype="float32").transpose(1, 2, 0)

            predict_mask = cv2.resize(predict, ori_image.size, interpolation=cv2.INTER_CUBIC)
            color = colorize(predict_mask.argmax(axis=2).astype(np.uint8), colors)
            color.save(os.path.join(args.draw_dir, os.path.basename(file_name)))
            # continue

            ########################################################################################################
            # eval
            label_path = os.path.join(args.gt_dir, file_name.replace("leftImg8bit.png", "gtFine_labelIds.png"))
            if not os.path.exists(label_path):
                continue
            
            # gt 
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, args.input_shape[::-1], interpolation=cv2.INTER_NEAREST)
            label = convert_label(label)
            target = np.array(label).astype('int32')
            target = np.expand_dims(target, 0)
            target = torch.from_numpy(target)

            confusion_matrix += get_confusion_matrix(
                target,
                tvm_predict,
                target.size(),
                num_classes,
                255)

            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)
            pixel_acc = tp.sum()/pos.sum()
            mean_acc = (tp/np.maximum(1.0, pos)).mean()
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            mean_IoU = IoU_array.mean()
            print("{:s}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(file_name, pixel_acc * 100, mean_IoU * 100))

            # ########################################################################################################


""" 

"""
