# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import numpy as np
import cv2
import os
import glob
import argparse
from degradations import GFPGAN_degradation
from tqdm import tqdm

def generate_degradation(val_dir):
    os.makedirs(os.path.join(val_dir, 'lq_'), exist_ok=True)

    hq_files = sorted(glob.glob(os.path.join(val_dir, 'hq', '*.jpg*')))
    print(len(hq_files),' in total')
    degrader = GFPGAN_degradation() 
    with tqdm(total=len(hq_files)) as pbar:
        for hq_f in hq_files:
            img_gt = cv2.imread(hq_f, cv2.IMREAD_COLOR)
            h, w = img_gt.shape[:2]
            img_gt = cv2.resize(img_gt, (h, w), interpolation=cv2.INTER_AREA)
            img_gt = img_gt.astype(np.float32)/255.
            img_gt, img_lq = degrader.degrade_process(img_gt)
            img_lq *= 255.0
            cv2.imwrite(os.path.join(val_dir, 'lq_', os.path.basename(hq_f)), img_lq)
            pbar.update(1) 


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', type=str, default='./code/eval/GPEN', help='input folder')

    args = parser.parse_args()
    generate_degradation(args.val_dir)
