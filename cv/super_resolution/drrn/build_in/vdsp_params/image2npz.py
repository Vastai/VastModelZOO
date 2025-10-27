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


def make_npz_text(args):
    os.makedirs(args.target_path, exist_ok=True)

    with open(args.text_path, 'w') as f:
        files_list = glob.glob(args.dataset_path + "/*.bmp")

        for img_file in tqdm(files_list):

            image = cv2.imread(img_file)
            image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            image_data = cv2.split(image_ycrcb)
            image_y_data = image_data[0] #/ 255. # vamp unsupport float data, using norma_type=DIV255
            data = np.stack((image_y_data, image_y_data, image_y_data))

            data_dict = {"input_0": data}
            save_path = os.path.join(args.target_path, os.path.splitext(os.path.basename(img_file))[0]+ ".npz")
            np.savez(save_path, **data_dict)
            f.write(os.path.abspath(save_path) + "\n")


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="MAKE DATA LIST")
    parse.add_argument("--dataset_path", type=str, default="/path/to/code/model_check/SR/sr/pytorch-vdsr/Set5_BMP/scale_4", help="path to input source image folder")
    parse.add_argument("--target_path", type=str, default="/path/to/code/model_check/SR/sr/pytorch-vdsr/Set5_BMP/scale_4_npz", help="path to output npz image folder")
    parse.add_argument("--text_path", type=str, default="npz_datalist.txt")
    args = parse.parse_args()
    print(args)

    make_npz_text(args)
