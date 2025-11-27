# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys
import glob
from tqdm import tqdm

from siglip_image_onnx import SiglipImageOnnx
current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

from siglip_model import SiglipImage
from siglip_model import NormalType

import vaststreamx as vsx
import cv2
import argparse
import numpy as np
from scipy import spatial
import onnxruntime as ort

def load_labels(file):
    with open(file) as fin:
        return [line.strip() for line in fin.readlines()]

def get_cosine(res_before, res_after, thresh_hold=1e-8):
    # print('res_before: {f}'.format(f=res_before.shape))
    # print('res_after: {f}'.format(f=res_after.shape))
    # print(f"res_before:{res_before}, res_after: {res_after}")
    if res_after is not None and res_before is not None:
        res_before = res_before.flatten().astype("float32")
        res_after = res_after.flatten().astype("float32")
        cos_sim_scipy =  1 - spatial.distance.cosine(res_before, res_after)
        # print('cos_sim:' + str(cos_sim_scipy))
        thresh_hold = thresh_hold
        # print(res_before.shape)
        # print(res_after.shape)
        try:
            np.testing.assert_allclose(res_before, res_after, atol=thresh_hold, rtol=thresh_hold)
            # return True
        except AssertionError as e:
            print(e)
        return cos_sim_scipy
    else:
        print('res_before or res_before is None!')
        print('res_before: {f}'.format(f=res_before))
        print('res_after: {f}'.format(f=res_after))

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/path/to/siglip_image-fp16-none-1_3_224_224-vacc/mod",
        help="image model prefix of the model suite files",
    )
    parser.add_argument(
        "--onnx_path",
        default="/path/to/siglip_image-fp16-none-1_3_224_224.onnx",
        help="image model onnx file",
    )
    parser.add_argument(
        "--hw_config",
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
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--input_file",
        default="",
        help="input file",
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="input dataset root",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    assert vsx.set_device(args.device_id) == 0
    model = SiglipImage(
        args.model_prefix,
        args.norm_elf,
        args.space2depth_elf,
        batch_size,
        args.device_id,
        NormalType.NORMAL_DIV255,
    )
    onnx_model = SiglipImageOnnx(args.onnx_path)

    if args.input_file != "":
        image = cv2.imread(args.input_file)
        assert image is not None, f"Failed to read input file: {args.input_file}"
        result_vsx = model.process(image)
        result_onnx = onnx_model.process(image)
        cos = get_cosine(result_vsx[0][0][0], result_onnx[0][0])
        print(f"score: {result_vsx[0]}, score_onnx: {result_onnx}, cos:{cos}")

    elif args.dataset_root != "":
        print(f"args.dataset_root:{args.dataset_root}")
        filelist = sorted(glob.glob(f"{args.dataset_root}/**/*.png", recursive=True)) + sorted(glob.glob(f"{args.dataset_root}/**/*.JPEG", recursive=True)) + sorted(glob.glob(f"{args.dataset_root}/**/*.jpg", recursive=True)) + sorted(glob.glob(f"{args.dataset_root}/**/*.jpeg", recursive=True))
        # filelist = [f for f in sorted(glob.glob(args.dataset_root + "/*.{png,jpg,jpeg}", recursive=True))]        
        cos_values = []
        print(f"filelist:{filelist}")
        for file in tqdm(filelist):
            image = cv2.imread(file)
            assert image is not None, f"Failed to read input file: {file}"
            result_vsx = model.process(image)
            result_onnx = onnx_model.process(image)
            cos = get_cosine(result_vsx[0][0][0], result_onnx[0][0])
            cos_values.append(cos)
        if cos_values:
            avg_cos = sum(cos_values) /len(cos_values)
            max_cos = max(cos_values)
            min_cos = min(cos_values)
            print(f"Average Cosine Similarity: {avg_cos}")
            print(f"Maximum Cosine Similarity: {max_cos}")
            print(f"Minimum Cosine Similarity: {min_cos}")
        else:
            print("No cosine similarity values were calculated.")
