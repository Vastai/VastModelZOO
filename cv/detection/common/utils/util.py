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
import numpy as np
import random
from typing import Optional, Union, Tuple, Iterator


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
    

class DataIterator(object):
    """Scan a directory to find the interested directories or files in
    arbitrary order.
    Note:
        :meth: returns the path relative to ``dir_path``.
    Args:
        dir_path (str | Path): Path of the directory.
        list_dir (bool): List the directories. Default: False.
        list_file (bool): List the path of files. Default: True.
        suffix (str or tuple[str], optional):  File suffix
            that we are interested in. Default: None.
        recursive (bool): If set to True, recursively scan the
            directory. Default: False.
        get_data_name (int or None): the number of how many data want to read.
    Yields:
        Iterable[str]: A relative path to ``dir_path``.
    """

    def __init__(
        self,
        dir_path: str,
        list_dir: bool = False,
        list_file: bool = True,
        suffix: Optional[Union[str, Tuple[str]]] = None,
        recursive: bool = True,
        get_data_num=None,
    ):
        self.get_data_num = get_data_num
        self.dir_path = dir_path
        self.list_dir = list_dir
        self.list_file = list_file
        self.suffix = suffix
        self.recursive = recursive
        self.files = self.get_files()

    def get_files(self):
        files = list(self.load_data())
        files_number = len(files)
        # get max number is files_number
        if self.get_data_num is None or self.get_data_num > files_number:
            self.get_data_num = files_number
        # logger.info(f"get files number is : {self.get_data_num}")
        return random.sample(files, self.get_data_num)

    def load_data(self) -> Iterator[str]:
        if self.list_dir and self.suffix is not None:
            raise TypeError("`suffix` should be None when `list_dir` is True")

        if (self.suffix is not None) and not isinstance(self.suffix, (str, tuple)):
            raise TypeError("`suffix` must be a string or tuple of strings")

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive):
            for entry in os.scandir(dir_path):
                if not entry.name.startswith(".") and entry.is_file():
                    if (suffix is None or entry.path.endswith(suffix)) and list_file:
                        yield entry.path
                elif os.path.isdir(entry.path):
                    if list_dir:
                        yield entry.path
                    if recursive:
                        yield from _list_dir_or_file(entry.path, list_dir, list_file, suffix, recursive)

        return _list_dir_or_file(self.dir_path, self.list_dir, self.list_file, self.suffix, self.recursive)


if __name__ == '__main__':
    mean = [1, 1, 1]
    for value in mean:
        print(half_to_uint16(value))
