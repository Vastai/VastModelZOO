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
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


def make_npz_text(args):
    os.makedirs(args.target_path, exist_ok=True)

    with open(args.text_path, 'w') as f:
        files_list = glob.glob(args.dataset_path + "/*.jpg")
        files_list.sort()
        for img_file in tqdm(files_list):
            image_src = Image.open(img_file)

            # 需要人脸配准，src，128-64-32-16，input
            pre_process = transforms.Compose([
                                            transforms.CenterCrop((178, 178)),
                                            transforms.Resize((128,128)),
                                                    ])
            input_image = pre_process(image_src)
            _16x16_down_sampling = transforms.Resize((16,16))
            _32x32_down_sampling = transforms.Resize((32, 32))
            _64x64_down_sampling = transforms.Resize((64, 64))

            # totensor = transforms.Compose([
            #                             transforms.ToTensor(),
            #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #                             ])

            #Note: Our netowork is trained by progressively downsampled images.
            transformed_image = _16x16_down_sampling(_32x32_down_sampling(_64x64_down_sampling(input_image)))
            # image_torch = totensor(transformed_image)
            data = np.array(transformed_image)
            data = data.transpose(2, 0, 1) # hwc--> chw

            data_dict = {"input_0": data}
            save_path = os.path.join(args.target_path, os.path.splitext(os.path.basename(img_file))[0]+ ".npz")
            np.savez(save_path, **data_dict)
            f.write(os.path.abspath(save_path) + "\n")


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="MAKE DATA LIST")
    parse.add_argument("--dataset_path", type=str, default="/path/to/dataset/sr/CelebA/Img/img_align_celeba_1k", help="path to input source image folder")
    parse.add_argument("--target_path", type=str, default="/path/to/dataset/sr/CelebA/Img/img_align_celeba_1k_npz", help="path to output npz image folder")
    parse.add_argument("--text_path", type=str, default="npz_datalist.txt")
    args = parse.parse_args()
    print(args)

    make_npz_text(args)
