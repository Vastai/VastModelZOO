
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
import shutil
import argparse
import numpy as np
from tqdm import tqdm


# 从JPEGImages中找出评估图像，并复制到一个新的文件夹，且至转换评估图像至npz
copy_val_dir = "./dataset/seg/VOCdevkit/VOC2012/JPEGImages_val"
val_text = "./dataset/seg/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"

with open(val_text, "r") as ff:
    val_list = ff.readlines()
val_list = [line.strip() for line in val_list]


def make_npz_text(args):
    os.makedirs(args.target_path, exist_ok=True)
    os.makedirs(copy_val_dir, exist_ok=True)

    with open(args.text_path, 'w') as f:
        files_list = glob.glob(args.dataset_path + "/*.jpg")

        for img_file in tqdm(files_list):
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            if img_name not in val_list:
                continue
            shutil.copy2(img_file, copy_val_dir)

            img_data = cv2.imread(img_file)
            img_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
            data = np.array(img_data)
            data = data.transpose(2,0,1) # hwc--> chw

            data_dict = {"input_0": data}
            save_path = os.path.join(args.target_path, os.path.splitext(os.path.basename(img_file))[0]+ ".npz")
            np.savez(save_path, **data_dict)
            f.write(os.path.abspath(save_path) + "\n")


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="MAKE DATA LIST")
    parse.add_argument("--dataset_path", type=str, default="./dataset/seg/VOCdevkit/VOC2012/JPEGImages", help="path to input source image folder")
    parse.add_argument("--target_path", type=str, default="./dataset/seg/VOCdevkit/VOC2012/JPEGImages_val_npz", help="path to output npz image folder")
    parse.add_argument("--text_path", type=str, default="npz_datalist.txt")
    args = parse.parse_args()
    print(args)

    make_npz_text(args)
