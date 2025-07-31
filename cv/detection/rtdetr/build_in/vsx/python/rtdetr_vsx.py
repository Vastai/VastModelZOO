# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import sys
from tqdm import tqdm

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

from common.rtdetr import RtDetrModel
import common.utils as utils

import cv2
import argparse
import glob


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix_path",
        default="deploy_weights/official_rtdetr_run_stream_fp16/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--vdsp_params_info",
        default="../vacc_code/rtdetr_vdsp_params.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--label_txt",
        default="data/labels/coco2id.txt",
        help="label file",
    )
    parser.add_argument(
        "--file_path",
        default="",
        help="dataset filename list",
    )
    parser.add_argument(
        "--save_dir",
        default="",
        help="dataset output folder path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    labels = utils.load_labels(args.label_txt)
    batch_size = 1
    model = RtDetrModel(args.model_prefix_path, args.vdsp_params_info, batch_size, args.device_id)
    threshold = 0.01
    model.set_threshold(threshold)
    image_format = model.get_fusion_op_iimage_format()

    if os.path.isfile(args.file_path):
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read {args.file_path}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        objects = model.process(vsx_image)
        print("Detection objects:")
        for obj in objects:
            if obj[1] >= 0:
                bbox = [int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])]
                print(
                    f"Object class: {labels[int(obj[0])]}, label:{int(obj[0])}, score: {obj[1]}, bbox: {bbox}"
                )

    else:
        filelist = glob.glob(os.path.join(args.file_path , "*"))
        for image_file in tqdm(filelist):
            fullname = image_file
            cv_image = cv2.imread(fullname)
            assert cv_image is not None, f"Failed to read {fullname}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            objects = model.process(vsx_image)
            base_name, _ = os.path.splitext(os.path.basename(image_file))
            outfile = open(
                os.path.join(args.save_dir, base_name + ".txt"), "wt"
            )
            #print(f"{image_file} detection objects:")
            for obj in objects:
                if obj[1] >= 0:
                    bbox = [int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])]
                    # print(
                    #     f"Object class: {labels[int(obj[0])]}, score: {obj[1]}, bbox: {bbox}"
                    # )
                    outfile.write(
                        f"{labels[int(obj[0])]} {obj[1]} {(obj[2]):.4f} {(obj[3]):.4f} {(obj[2]+obj[4]):.4f} {(obj[3]+obj[5]):.4f}\n"
                    )
                else:
                    break
            outfile.close()
