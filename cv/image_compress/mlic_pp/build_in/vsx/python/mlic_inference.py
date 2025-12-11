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
import vaststreamx as vsx

import cv2
import argparse
from mlic_base import MLICPlusPlus
import torchvision
import torch
import math
import time
import numpy as np
from torch.nn import functional as F

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
        "--gaha_model_prefix",
        default="deploy_weights/compress_ga_ha_sim_512_768_run_stream_fp16/mod",
        help="ga_ha model prefix of the model suite files",
    )
    parser.add_argument(
        "--gaha_hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--gaha_vdsp_params",
        default=" ../build_in/vdsp_params/mlic_compress_gaha_rgbplanar.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--hs_model_prefix",
        default="deploy_weights/compress_hs_sim_run_stream_fp16/mod",
        help="h_s model prefix of the model suite files",
    )
    parser.add_argument(
        "--hs_hw_config",
        default="",
        help="hs_hw-config file of the model suite",
    )
    parser.add_argument(
        "--gs_model_prefix",
        default="deploy_weights/decompress_gs_sim_run_stream_fp16/mod",
        help="g_s model prefix of the model suite files",
    )
    parser.add_argument(
        "--gs_hw_config",
        default="",
        help="gs_hw-config file of the model suite",
    )
    parser.add_argument(
        "--torch_model",
        default="/path/to/mlicpp_mse_q5_2960000.pth.tar",
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
        default="./result.jpg",
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
        default=64,
        help="padding patch size (default: %(default)s)",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argument_parser()
    torch.backends.cudnn.deterministic = True
    batch_size = 1

    mlic = MLICPlusPlus(
        args.gaha_model_prefix,
        args.gaha_vdsp_params,
        args.hs_model_prefix,
        args.gs_model_prefix,
        args.tensorize_elf_path,
        batch_size,
        args.device_id,
        args.gaha_hw_config,
        args.hs_hw_config,
        args.gs_hw_config,
        args.patch,
    )
    checkpoint = torch.load(
        args.torch_model, map_location=torch.device("cpu"), weights_only=True
    )
    mlic.load_state_dict(checkpoint["state_dict"])

    image_format = mlic.get_fusion_op_iimage_format()

    # disable graph building and gradient computation
    mlic.eval()
    torch.set_grad_enabled(False)

    if args.dataset_path == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Read image failed:{args.input_file}"
        # vacc
        vsx_image = cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)

        com_out = mlic.compress(vsx_image)
        decom_out = mlic.decompress(com_out["strings"], com_out["shape"])

        decom_out["x_hat"] = torch.nn.functional.pad(
            decom_out["x_hat"],
            (0, -com_out["params"][3], 0, -com_out["params"][2]),
        )
        torchvision.utils.save_image(decom_out["x_hat"], args.output_file, nrow=1)
        print(
            "compress time: ",
            com_out["cost_time"],
            "decompress time: ",
            decom_out["cost_time"],
        )
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

            com_out = mlic.compress(vsx_image)
            # print(f"    Compress time:{com_out['cost_time']*1000} ms")
            decom_out = mlic.decompress(com_out["strings"], com_out["shape"])
            # print(f"    Decompress time:{decom_out['cost_time']*1000} ms")

            decom_out["x_hat"] = torch.nn.functional.pad(
                decom_out["x_hat"],
                (0, -com_out["params"][3], 0, -com_out["params"][2]),
            )

            if (
                cv_image.shape[0] != decom_out["x_hat"].shape[2]
                or cv_image.shape[1] != decom_out["x_hat"].shape[3]
            ):
                cv_image = cv2.resize(
                    cv_image,
                    (decom_out["x_hat"].shape[3], decom_out["x_hat"].shape[2]),
                    interpolation=cv2.INTER_LINEAR,
                )
            cv_image = np.array(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)).transpose(
                2, 0, 1
            )
            cv_image = cv_image[np.newaxis, :] / 255.0
            psnr = compute_psnr(decom_out["x_hat"], torch.from_numpy(cv_image))
            print(f"psnr:{psnr}")
            pnsrs.append(psnr)
            compress_times.append(com_out["cost_time"])
            decompress_times.append(decom_out["cost_time"])

            out_file = os.path.join(args.dataset_output_path, os.path.basename(file))
            torchvision.utils.save_image(decom_out["x_hat"], out_file, nrow=1)

        average_compress_time = sum(compress_times) / float(len(compress_times))
        average_decompress_time = sum(decompress_times) / float(len(decompress_times))
        average_pnsr = sum(pnsrs) / float(len(pnsrs))
        print(f"    Ave Compress time:{average_compress_time*1000} ms")
        print(f"    Ave Decompress time:{average_decompress_time*1000} ms")
        print(f"    Ave PNSR:{average_pnsr}")
