
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
        files_list = glob.glob(args.dataset_path + "/*.jpg")

        for img_file in tqdm(files_list):

            img_data = cv2.imread(img_file)
            img_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
            data = np.array(img_data)
            data = data.transpose(2, 0, 1) # hwc--> chw

            data_dict = {"input_0": data}
            save_path = os.path.join(args.target_path, os.path.splitext(os.path.basename(img_file))[0]+ ".npz")
            np.savez(save_path, **data_dict)
            f.write(os.path.abspath(save_path) + "\n")


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="MAKE DATA LIST")
    parse.add_argument("--dataset_path", type=str, default="/path/to/dataset/seg/carvana/imgs", help="path to input source image folder")
    parse.add_argument("--target_path", type=str, default="/path/to/dataset/seg/carvana/imgs_npz", help="path to output npz image folder")
    parse.add_argument("--text_path", type=str, default="npz_datalist.txt")
    args = parse.parse_args()
    print(args)

    make_npz_text(args)
