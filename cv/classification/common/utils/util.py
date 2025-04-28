# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :    tonyx
@Email  :    algorithm@vastaitech.com
@Time   :    2025/04/27 10:17:38
'''

import os
import cv2
import glob
import numpy as np


def get_activation_aligned_faster(activation, dtype=np.float16, fc_mode=False, force_int8_layout_to_fp16=False):  # NCHW
    N = C = H = W = 1
    if len(activation.shape) == 2:
        N, C = activation.shape
        fc_mode = True
    elif len(activation.shape) == 5:
        N, C, H, W, B = activation.shape
    elif len(activation.shape) == 1:
        (C,) = activation.shape
    else:
        N, C, H, W = activation.shape
    h_group = w_group = c_group = 0
    if H == 1 and W == 1 and fc_mode == True:
        if dtype == np.float16:
            h_group, w_group, c_group = 1, 1, 256
        elif dtype == np.int8:
            h_group, w_group, c_group = 1, 1, 512
    else:
        if dtype == np.float16 or force_int8_layout_to_fp16:
            h_group, w_group, c_group = 8, 8, 4
        elif dtype == np.int8:
            h_group, w_group, c_group = 8, 8, 8
    pad_H, pad_W, pad_C = H, W, C
    pad_h, pad_w = 0, 0
    if H % h_group != 0:
        pad_h = h_group - H % h_group
        pad_H += pad_h
    if W % w_group != 0:
        pad_w = w_group - W % w_group
        pad_W += pad_w
    if C % c_group != 0:
        pad_c = c_group - C % c_group
        pad_C += pad_c
    # tensorize to WHC4c8h8w
    w_num = pad_W // w_group
    h_num = pad_H // h_group
    c_num = pad_C // c_group
    n_num = N
    block_size = w_group * h_group * c_group
    activation = activation.astype(dtype)
    assert(len(activation.shape) == 4)
    if (pad_h | pad_w) != 0:
        activation = np.pad(activation, ((0,0),(0,0),(0,pad_h),(0,pad_w)))
    np_arr = np.zeros((n_num, w_num, h_num, c_num, block_size), dtype)
    block_size_hacked = 3 * 8 * 8
    c_group_hacked = 3
    for n in range(N):
        for c in range(c_num):
            c_index = c * c_group_hacked
            for h in range(h_num):
                h_index = h * h_group
                for w in range(w_num):
                    w_index = w * w_group
                    # print(activation[n, c_index:c_index+c_group_hacked, h_index:h_index+h_group, w_index:w_index+w_group].shape)
                    np_arr[n, w, h, c, :block_size_hacked] = activation[n, c_index:c_index+c_group_hacked, h_index:h_index+h_group, w_index:w_index+w_group].flatten()
    return np_arr


def get_activation_aligned(activation, dtype=np.float16, fc_mode=False, force_int8_layout_to_fp16=False):  # NCHW
    N = C = H = W = 1
    if len(activation.shape) == 2:
        N, C = activation.shape
        fc_mode = True
    elif len(activation.shape) == 5:
        N, C, H, W, B = activation.shape
    elif len(activation.shape) == 1:
        (C,) = activation.shape
    else:
        N, C, H, W = activation.shape
    h_group = w_group = c_group = 0
    if H == 1 and W == 1 and fc_mode == True:
        if dtype == np.float16:
            h_group, w_group, c_group = 1, 1, 256
        elif dtype == np.int8:
            h_group, w_group, c_group = 1, 1, 512
    else:
        if dtype == np.float16 or force_int8_layout_to_fp16:
            h_group, w_group, c_group = 8, 8, 4
        elif dtype == np.int8:
            h_group, w_group, c_group = 8, 8, 8
    pad_H, pad_W, pad_C = H, W, C
    if H % h_group != 0:
        pad_h = h_group - H % h_group
        pad_H += pad_h
    if W % w_group != 0:
        pad_w = w_group - W % w_group
        pad_W += pad_w
    if C % c_group != 0:
        pad_c = c_group - C % c_group
        pad_C += pad_c
    # tensorize to WHC4c8h8w
    w_num = pad_W // w_group
    h_num = pad_H // h_group
    c_num = pad_C // c_group
    n_num = N
    block_size = w_group * h_group * c_group
    activation = activation.astype(dtype)
    np_arr = np.zeros((n_num, w_num, h_num, c_num, block_size), dtype)
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    addr = (c % c_group) * h_group * w_group + (h % h_group) * w_group + (w % w_group)
                    if len(activation.shape) == 2:
                        np_arr[n, w // w_group, h // h_group, c // c_group, addr] = activation[n, c]
                    elif len(activation.shape) == 1:
                        np_arr[n, w // w_group, h // h_group, c // c_group, addr] = activation[n]
                    else:
                        np_arr[n, w // w_group, h // h_group, c // c_group, addr] = activation[n, c, h, w]
    return np_arr


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

def convert_to_rgb_planar(image_file):
    image_bgr = cv2.imread(image_file, cv2.IMREAD_COLOR)
    assert len(image_bgr.shape) == 3
    height, width, c = image_bgr.shape
    assert c == 3
    b, g, r = cv2.split(image_bgr)
    image_rgb_planar = np.stack((r, g, b))

    return image_rgb_planar, height, width


def convert_rgb_to_yuv(image_file):
    image_bgr = cv2.imread(image_file, cv2.IMREAD_COLOR)
    image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV_I420)

    return image_yuv
    


if __name__ == '__main__':
    mean = [1, 1, 1]
    for value in mean:
        print(half_to_uint16(value))
