import cv2
import os
import glob
import argparse
import copy
import numpy as np
from tqdm import tqdm


def generate_degradation(val_dir):
    save_dir = os.path.join(val_dir, '..', 'hq_gray')
    os.makedirs(save_dir, exist_ok=True)

    hq_files = sorted(glob.glob(os.path.join(val_dir, '*.jpg*')))
    print(len(hq_files),' in total')
    with tqdm(total=len(hq_files)) as pbar:
        for hq_f in hq_files:
            img_gt = cv2.imread(hq_f, cv2.IMREAD_COLOR)
            img_gt = cv2.resize(img_gt, (512, 512), interpolation=cv2.INTER_LINEAR)
            img_lq = copy.deepcopy(img_gt)
            temp = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            for i in range(3):
                img_lq[:, :, i] = temp
            cv2.imwrite(os.path.join(save_dir, os.path.basename(hq_f)), img_lq)
            pbar.update(1) 


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', type=str, default='/path/to/dataset/face/GPEN/hq', help='input folder')

    args = parser.parse_args()
    generate_degradation(args.val_dir)
