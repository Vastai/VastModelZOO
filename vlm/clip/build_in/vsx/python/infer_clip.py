# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys


from clip_model import ClipModel
import glob
import vaststreamx as vsx
import cv2
import argparse
import numpy as np
from tqdm import tqdm


def load_labels(file):
    with open(file) as fin:
        return [line.strip() for line in fin.readlines()]

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

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgmod_prefix",
        default="/path/to/clip_image_openai_run_stream_fp16/mod",
        help="image model prefix of the model suite files",
    )
    parser.add_argument(
        "--imgmod_hw_config",
        help="image model hw-config file of the model suite",
        default="",
    )
    parser.add_argument(
        "--norm_elf",
        default="/path/to/elf/normalize",
        help="image model elf file",
    )
    parser.add_argument(
        "--space2depth_elf",
        default="/path/to/elf/space_to_depth",
        help="image model elf file",
    )
    parser.add_argument(
        "--txtmod_prefix",
        default="/path/to/clip_text_openai_run_stream_fp16/resnet50",
        help="text model prefix of the model suite files",
    )
    parser.add_argument(
        "--txtmod_hw_config",
        help="text model hw-config file of the model suite",
        default="",
    )
    parser.add_argument(
        "--txtmod_vdsp_params",
        default="../vacc_code/vdsp_params/openai-clip-vdsp_params.json",
        help="text model vdsp preprocess parameter file",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--file_path",
        default="data/images/cat.jpg",
        help="input file",
    )
    parser.add_argument(
        "--label_file",
        default="data/labels/imagenet.txt",
        help="label file",
    )
    parser.add_argument(
        "--dataset_output_file",
        default="",
        help="dataset output file",
    )
    parser.add_argument(
        "--strings",
        default="[a diagram,a dog,a cat]",
        help="test strings, split by \",\"",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    labels = load_labels(args.label_file)
    batch_size = 1
    assert vsx.set_device(args.device_id) == 0
    model = ClipModel(
        args.imgmod_prefix,
        args.norm_elf,
        args.space2depth_elf,
        args.txtmod_prefix,
        args.txtmod_vdsp_params,
        batch_size,
        args.device_id,
    )

    if os.path.isfile(args.file_path):
        image = cv2.imread(args.file_path)
        assert image is not None, f"Failed to read input file: {args.input_file}"
        texts = args.strings.strip('[').strip(']').split(',')
        print(f"intput texts:{texts}")

        result = model.process(image=image, texts=texts)
        index = np.argsort(result)[::-1]
        print("Top5:")
        for i in range(5):
            print(f"{i}th, class name: {labels[index[i]]}, score: {result[index[i]]}")
    else:
        texts_features = model.process_texts(labels)
        filelist, _ = find_file_by_suffix(args.file_path, suffix_list=[".JPEG"], recursive=True)
        with open(args.dataset_output_file, "wt") as fout:
            for file in tqdm(filelist):
                fullname = file
                # print(fullname)
                image = cv2.imread(fullname)
                assert image is not None, f"Failed to read input file: {fullname}"
                image_feature = model.process_image(image)
                result = model.post_process(image_feature, texts_features)
                index = np.argsort(result)[::-1]
                for i in range(5):
                    fout.write(
                        f"{file}: top-{i} id: {index[i]}, prob: {result[index[i]]}, class name: {labels[index[i]]}\n"
                    )


# [VACC]:  top1_rate: 54.8 top5_rate: 82.3