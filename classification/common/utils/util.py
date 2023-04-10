# ==============================================================================
# 
# Copyright (C) 2023 VastaiTech Technologies Inc.  All rights reserved.
# 
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :     vastml
@Time  : 2023/04/07 22:04:53
'''

import glob
import os
import numpy as np


def uint16_to_half(x):
    """Convert a uint16 represented number to np.float16
    """
    return np.frombuffer(np.array(x, dtype=np.uint16), dtype=np.float16)[0]


def half_to_uint16(x):
    """Convert a np.float16 number to a uint16 represented
    """
    return int(np.frombuffer(np.array(x, dtype=np.float16), dtype=np.uint16))


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



if __name__ == '__main__':
    mean = [1, 1, 1]
    for value in mean:
        print(half_to_uint16(value))
