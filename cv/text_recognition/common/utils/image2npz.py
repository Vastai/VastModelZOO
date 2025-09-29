# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import argparse
import os

import cv2
import glob
import numpy as np
from tqdm import tqdm
def find_file_by_suffix(data_dir, suffix_list=None, recursive=False):
    """
    Based on the given folder and suffix list, find all files in the suffix list under the folder.
    Args:
        data_dir: given reference suffix, e.g. [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".pgm"]
        suffix_list: sub folder recursive or not
        recursive: a list of file paths within a folder that match the suffix
    Returns:
        find_suffix_list: list of suffixes in the folder
    """
    if suffix_list is None:
        suffix_list = [".jpg", ".jpeg", ".png", ".bmp"]

    find_file_list = []
    find_suffix_list = []
    for suffix in suffix_list:
        if recursive:
            find_files = glob.glob(os.path.join(data_dir, "**/*" + suffix), recursive=recursive)
        else:
            find_files = glob.glob(os.path.join(data_dir, "*" + suffix))
        find_file_list.extend(find_files)
        if find_files:
            find_suffix_list.append(suffix)

    return find_file_list, find_suffix_list


parse = argparse.ArgumentParser(description="MAKE DATA LIST")
parse.add_argument("--dataset_path", type=str)
parse.add_argument("--target_path", type=str)
parse.add_argument("--text_path", type=str, default="npz.txt")
args = parse.parse_args()
print(args)


def make_npz_text(args):
    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path, exist_ok=True)
    with open(args.text_path, 'w') as f:
        files_list, _ = find_file_by_suffix(args.dataset_path, suffix_list=[".jpg"], recursive=True)

        for img_file in tqdm(files_list):
            sub_folder =  os.path.split(img_file)[0].split("/")[-1]
            os.makedirs(os.path.join(args.target_path, sub_folder), exist_ok=True)

            img_data = cv2.imread(img_file)
            img_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
            data = np.array(img_data)
            data = data.transpose(2,0,1) # hwc--> chw

            data_dict = {"input_0": data}
            save_path = os.path.join(args.target_path, sub_folder, os.path.splitext(os.path.split(img_file)[-1])[0]+ ".npz")
            np.savez(save_path, **data_dict)
            f.write(os.path.abspath(save_path) + "\n")


if __name__ == "__main__":
    make_npz_text(args)
