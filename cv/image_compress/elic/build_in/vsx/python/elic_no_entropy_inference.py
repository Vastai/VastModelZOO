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
from elic_base import VastaiElicNoEntropy
import torchvision
import torch
import math
import time
import numpy as np
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
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/elic-noentropy/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="./data/configs/elic_compress_gaha_rgb888.json",
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

    elic_noentropy = VastaiElicNoEntropy(
        args.model_prefix,
        args.vdsp_params,
        batch_size,
        args.device_id,
        args.hw_config,
        args.patch,
    )
    image_format = elic_noentropy.get_fusion_op_iimage_format()

    if args.dataset_path == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Read image failed:{args.input_file}"
        vsx_image = cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)

        h = vsx_image.height
        w = vsx_image.width
        p = args.patch
        new_h = (h + p - 1) // p * p
        new_w = (w + p - 1) // p * p
        padding_left = (new_w - w) // 2
        padding_right = new_w - w - padding_left
        padding_top = (new_h - h) // 2
        padding_bottom = new_h - h - padding_top

        com_out = elic_noentropy.inference(vsx_image)
        com_out["x_hat"] = torch.nn.functional.pad(
            com_out["x_hat"],
            (-padding_left, -padding_right, -padding_top, -padding_bottom),
        )
        torchvision.utils.save_image(com_out["x_hat"], args.output_file, nrow=1)
    else:
        filepaths = collect_images(args.dataset_path)
        filepaths = sorted(filepaths)
        if len(filepaths) == 0:
            print(
                f"Error: no images found in directory:{args.dataset_path}.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.dataset_output_path != "":
            os.makedirs(args.dataset_output_path, exist_ok=True)
        times = []
        decompress_times = []
        pnsrs = []
        bbps = []

        for file in filepaths:
            print(f"image file:{file}")
            cv_image = cv2.imread(file)
            assert cv_image is not None, f"Read image failed:{file}"
            vsx_image = cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )

            h = vsx_image.height
            w = vsx_image.width
            p = args.patch
            new_h = (h + p - 1) // p * p
            new_w = (w + p - 1) // p * p
            padding_left = (new_w - w) // 2
            padding_right = new_w - w - padding_left
            padding_top = (new_h - h) // 2
            padding_bottom = new_h - h - padding_top

            start = time.time()
            com_out = elic_noentropy.inference(vsx_image)
            times.append(time.time() - start)

            com_out["x_hat"] = torch.nn.functional.pad(
                com_out["x_hat"],
                (-padding_left, -padding_right, -padding_top, -padding_bottom),
            )

            num_pixels = vsx_image.height * vsx_image.width
            # 需要将likelihoods_y和likelihoods_z输出中0替换为极小值，代码如下：

            bpp = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in com_out["likelihoods"].values()
            )
            y_bpp = torch.log(com_out["likelihoods"]["y"]).sum() / (
                -math.log(2) * num_pixels
            )
            z_bpp = torch.log(com_out["likelihoods"]["z"]).sum() / (
                -math.log(2) * num_pixels
            )

            if args.dataset_output_path != "":
                out_file = os.path.join(
                    args.dataset_output_path, os.path.basename(file)
                )
                torchvision.utils.save_image(com_out["x_hat"], out_file, nrow=1)

            cv_image = np.array(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)).transpose(
                2, 0, 1
            )
            cv_image = cv_image[np.newaxis, :] / 255.0

            psnr = compute_psnr(com_out["x_hat"], torch.from_numpy(cv_image))
            print(f"psnr:{psnr}, bpp:{bpp.item()}")
            # exit(0)
            pnsrs.append(psnr)
            bbps.append(bpp.item())

        average_noentropy_times = sum(times) / float(len(times))
        average_pnsr = sum(pnsrs) / float(len(pnsrs))
        average_bpp = sum(bbps) / float(len(bbps))
        print(f"    Ave Compress time:{average_noentropy_times*1000} ms")
        print(f"    Ave PNSR:{average_pnsr}")
        print(f"    Ave bbp:{average_bpp}")
