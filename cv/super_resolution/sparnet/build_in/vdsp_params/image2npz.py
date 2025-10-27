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
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


def make_npz_text(args):
    os.makedirs(args.lr_path, exist_ok=True)
    os.makedirs(args.target_path, exist_ok=True)

    with open(args.text_path, 'w') as f:
        files_list = glob.glob(args.hr_path + "/*.png")
        files_list.sort()
        for img_file in tqdm(files_list):
            image_src = Image.open(img_file)

            input_image = image_src.resize((16, 16), resample=Image.BICUBIC)
            input_image.save(os.path.join(args.lr_path, os.path.basename(img_file)))

            data = np.array(input_image)
            data = data.transpose(2, 0, 1) # hwc--> chw

            data_dict = {"input_0": data}
            save_path = os.path.join(args.target_path, os.path.splitext(os.path.basename(img_file))[0]+ ".npz")
            np.savez(save_path, **data_dict)
            f.write(os.path.abspath(save_path) + "\n")


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="MAKE DATA LIST")
    parse.add_argument("--hr_path", type=str, default="./code/model_check/SR/face/Face-SPARNet/datasets/test_dirs/Helen_test_DIC/HR", help="path to input source image folder")
    parse.add_argument("--lr_path", type=str, default="./code/model_check/SR/face/Face-SPARNet/datasets/test_dirs/Helen_test_DIC/LR", help="path to input source image folder")
    parse.add_argument("--target_path", type=str, default="./code/model_check/SR/face/Face-SPARNet/datasets/test_dirs/Helen_test_DIC/LR_npz", help="path to output npz image folder")
    parse.add_argument("--text_path", type=str, default="npz_datalist.txt")
    args = parse.parse_args()
    print(args)

    make_npz_text(args)
