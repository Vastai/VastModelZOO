# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import sys
from typing import List

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

import cv2
import argparse
from dynamic_elic_compress import DynamicElicCompress
from dynamic_elic_decompress import DynamicElicDecompress
import torchvision
import torch
import math
import time
import numpy as np
import ast
import vaststreamx as vsx

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

def cv_bgr888_to_nv12(bgr888):
    yuv_image = cv2.cvtColor(bgr888, cv2.COLOR_BGR2YUV_I420)
    height, width = bgr888.shape[:2]
    y = yuv_image[:height, :]
    u = yuv_image[height : height + height // 4, :]
    v = yuv_image[height + height // 4 :, :]
    u = np.reshape(u, (height // 2, width // 2))
    v = np.reshape(v, (height // 2, width // 2))
    uv_plane = np.empty((height // 2, width), dtype=np.uint8)
    uv_plane[:, 0::2] = u
    uv_plane[:, 1::2] = v
    yuv_nv12 = np.concatenate((y, uv_plane), axis=0)
    return yuv_nv12

def cv_bgr888_to_vsximage(bgr888, vsx_format, device_id):
    h, w = bgr888.shape[:2]
    if vsx_format == vsx.ImageFormat.BGR_INTERLEAVE:
        res = bgr888
    elif vsx_format == vsx.ImageFormat.BGR_PLANAR:
        res = np.array(bgr888).transpose(2, 0, 1)
    elif vsx_format == vsx.ImageFormat.RGB_INTERLEAVE:
        res = cv2.cvtColor(bgr888, cv2.COLOR_BGR2RGB)
    elif vsx_format == vsx.ImageFormat.RGB_PLANAR:
        res = np.array(cv2.cvtColor(bgr888, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    elif vsx_format == vsx.ImageFormat.YUV_NV12:
        res = cv_bgr888_to_nv12(bgr888=bgr888)
    else:
        assert False, f"Unsupport format:{vsx_format}"
    return vsx.create_image(
        res,
        vsx_format,
        w,
        h,
        device_id,
    )


def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gaha_model_info",
        default="/opt/vastai/vaststreamx/data/models/elic-compress-g_a-h_a-fp16-1_3_512_512-vacc/mod_info.json",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--gaha_hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--gaha_vdsp_params",
        default="./data/configs/elic_compress_gaha_rgb888.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--max_input_shape",
        default="[1,3,1024,1024]",
        help="model max input shape, max supported shape: [1,3,1024,1024]",
    )
    parser.add_argument(
        "--hs_model_info",
        default="/opt/vastai/vaststreamx/data/models/elic-compress-h_s_chunk-fp16-1_192_8_8-vacc/mod_info.json",
        help="h_s model prefix of the model suite files",
    )
    parser.add_argument(
        "--hs_hw_config",
        default="",
        help="hs_hw-config file of the model suite",
    )
    parser.add_argument(
        "--gs0_model_info",
        default="/opt/vastai/vaststreamx/data/models/elic-compress-g_s_chunk-fp16-1_192_8_8-vacc/mod_info.json",
        help="g_s model prefix of the model suite files",
    )
    parser.add_argument(
        "--gs_model_info",
        default="/opt/vastai/vaststreamx/data/models/elic-compress-g_s_chunk-fp16-1_192_8_8-vacc/mod_info.json",
        help="g_s model prefix of the model suite files",
    )
    parser.add_argument(
        "--gs0_hw_config",
        default="",
        help="gs0_hw-config file of the model suite",
    )
    parser.add_argument(
        "--gs_hw_config",
        default="",
        help="gs_hw-config file of the model suite",
    )
    parser.add_argument(
        "--torch_model",
        default="",
        help="torch model file",
    )
    parser.add_argument(
        "--tensorize_elf_path",
        default="",
        help="tensorize elf file",
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
        default="../../../data/images/cycling.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="./elic_compress_result.jpg",
        help="output file",
    )
    parser.add_argument(
        "--dataset_path",
        default="",
        help="input dataset path",
    )
    parser.add_argument(
        "--dataset_output_path",
        default="",
        help="dataset output path",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=256,
        help="padding patch size (default: %(default)s)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    max_input_shape = ast.literal_eval(args.max_input_shape)
    print(f"max_input_shape:{max_input_shape}")
    compressor = DynamicElicCompress(
        args.gaha_model_info,
        args.gaha_vdsp_params,
        [max_input_shape],
        args.hs_model_info,
        args.torch_model,
        batch_size,
        args.device_id,
        args.gaha_hw_config,
        args.hs_hw_config,
        args.patch,
    )
    image_format = compressor.get_fusion_op_iimage_format()

    decompressor = DynamicElicDecompress(
        args.hs_model_info,
        args.gs0_model_info,
        args.gs_model_info,
        args.torch_model,
        args.tensorize_elf_path,
        batch_size,
        args.device_id,
        args.hs_hw_config,
        args.gs0_hw_config,
        args.gs_hw_config,
    )
    if args.dataset_path == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Read image failed:{args.input_file}"
        vsx_image = cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)

        p = args.patch
        h = vsx_image.height
        w = vsx_image.width
        new_h = (h + p - 1) // p * p
        new_w = (w + p - 1) // p * p
        padding_left = 0
        padding_right = new_w - w - padding_left
        padding_top = 0
        padding_bottom = new_h - h - padding_top
        com_out = compressor.process(vsx_image)
        decom_out = decompressor.decompress(com_out["strings"], com_out["shape"])
        decom_out["x_hat"] = torch.nn.functional.pad(
            decom_out["x_hat"],
            (-padding_left, -padding_right, -padding_top, -padding_bottom),
        )
        torchvision.utils.save_image(decom_out["x_hat"], args.output_file, nrow=1)
    else:
        filepaths = collect_images(args.dataset_path)
        filepaths = sorted(filepaths)
        if len(filepaths) == 0:
            print(
                f"Error: no images found in directory:{args.dataset_path}.",
                file=sys.stderr,
            )
            sys.exit(1)
        os.makedirs(args.dataset_output_path, exist_ok=True)
        compress_times = []
        decompress_times = []
        pnsrs = []

        for file in filepaths:
            print(f"image file:{file}")
            cv_image = cv2.imread(file)
            assert cv_image is not None, f"Read image failed:{file}"
            vsx_image = cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )

            start = time.time()
            com_out = compressor.process(vsx_image)
            compress_times.append(time.time() - start)

            p = args.patch
            h = vsx_image.height
            w = vsx_image.width
            new_h = (h + p - 1) // p * p
            new_w = (w + p - 1) // p * p
            padding_left = 0
            padding_right = new_w - w - padding_left
            padding_top = 0
            padding_bottom = new_h - h - padding_top

            start = time.time()
            decom_out = decompressor.decompress(com_out["strings"], com_out["shape"])
            decompress_times.append(time.time() - start)
            decom_out["x_hat"] = torch.nn.functional.pad(
                decom_out["x_hat"],
                (-padding_left, -padding_right, -padding_top, -padding_bottom),
            )

            cv_image = np.array(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)).transpose(
                2, 0, 1
            )
            cv_image = cv_image[np.newaxis, :] / 255.0

            psnr = compute_psnr(decom_out["x_hat"], torch.from_numpy(cv_image))
            print(f"psnr:{psnr}")
            pnsrs.append(psnr)

            out_file = os.path.join(args.dataset_output_path, os.path.basename(file))
            torchvision.utils.save_image(decom_out["x_hat"], out_file, nrow=1)

        average_compress_time = sum(compress_times) / float(len(compress_times))
        average_decompress_time = sum(decompress_times) / float(len(decompress_times))
        average_pnsr = sum(pnsrs) / float(len(pnsrs))
        print(f"    Ave Compress time:{average_compress_time*1000} ms")
        print(f"    Ave Decompress time:{average_decompress_time*1000} ms")
        print(f"    Ave PNSR:{average_pnsr}")
