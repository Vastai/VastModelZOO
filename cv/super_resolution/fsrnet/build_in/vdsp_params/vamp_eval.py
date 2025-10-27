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
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

def _denormalize(tensor):
    # (N,C,W,H) >> (N,W,H,C)
    #   [-1, 1] >> [0, 255]
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    image = np.clip(image, 0, 255)
    return image


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--gt_dir", type=str, default="/path/to/dataset/sr/CelebAMask-HQ/test_img", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="/path/to/code/vamc/vamp/0.2.0/outputs/fsrnet", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[128, 128], help="vamp input shape")
    parse.add_argument("--draw_dir", type=str, default="./npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()
    
    psnr_list = []
    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        psnr_list = []
        ssim_list = []
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".jpg")

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            heatmap = np.load(npz_file, allow_pickle=True)["output_2"].astype(np.float32)
            
            # draw
            out = _denormalize(torch.from_numpy(heatmap))[0].astype('uint8')
            if out.shape[0] != args.input_shape:
                out = cv2.resize(out, args.input_shape)
            cv2.imwrite(os.path.join(args.draw_dir, file_name), out[:,:,::-1])

            # eval
            from skimage.metrics import peak_signal_noise_ratio as compare_psnr
            from skimage.metrics import structural_similarity as compare_ssim
            image_hr =  Image.open(os.path.join(args.gt_dir, file_name))
            image_hr = image_hr.resize(args.input_shape, Image.BICUBIC)
            image_hr = np.array(image_hr)

            psnr = compare_psnr(image_hr, out)
            ssim = compare_ssim(image_hr, out, multichannel=True)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            print("{} ---> psnr:{}, ssim:{}".format(file_name, psnr, ssim))

        print("mean psnr:{}, mean ssim:{}".format(np.mean(psnr_list), np.mean(ssim_list)))

"""
fsrnet-fp16-none-3_128_128-debug
mean psnr:18.920412738618584, mean ssim:0.5815203089531279
fsrnet-int8-kl_divergence-3_128_128-debug
mean psnr:15.339090593746706, mean ssim:0.32787449184025896
"""