# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import argparse
import numpy as np
import glob
import os

from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.integrate import simps

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')


def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark 
    """
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = 34  # meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt,
                                        axis=1)) / (interocular * L)

    return rmse


def compute_auc(errors, failureThreshold, step=0.0001, showCurve=False):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()

    return AUC, failureRate

def load_gt(gt_files: str):
    gt_landmark = {}
    with open(gt_files, 'r') as fr:
        for line in fr:
            tmp = line.strip().split()
            img_name = os.path.basename(tmp[0]).split('.')[0]
            gt_landmark[img_name] =  np.expand_dims(np.asarray(tmp[1:197], dtype=np.float32), 0)
    return gt_landmark


def main(args):
    npz_files = glob.glob(args.result + "/*.npz")
    npz_files.sort()
    img_list = []
    nme_list = []

    if not args.debug:
        with open(args.npz_txt, 'r') as fr:
            for line in fr:
                img_list.append(os.path.basename(line.strip()).split('.')[0])
    gt_landmark_dict = load_gt(args.gt)

    if args.show_image:
        import shutil
        if os.path.exists('show_images'):
            shutil.rmtree('show_images')
        os.makedirs('show_images')

    for index, file in enumerate(tqdm(npz_files)):
        ## pred landmark
        ## output0在测试时不用
        landmarks = np.load(file, allow_pickle=True)["output_1"]
        landmarks = landmarks.reshape(landmarks.shape[0], -1,
                                        2)  

        ## landmark_gt
        if args.debug:
            landmark_gt = gt_landmark_dict[os.path.basename(file).split('.')[0]]
        else:
            landmark_gt = gt_landmark_dict[img_list[index]]

        landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1,
                                            2)

        if args.show_image:
            import cv2
            img = cv2.imread(os.path.join(args.images, img_list[index]+'.png'))
            pre_landmark = landmarks[0] * [112, 112]
            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

            cv2.imwrite(os.path.join('show_images', img_list[index]+'.png'), img)

        nme_temp = compute_nme(landmarks, landmark_gt)
        for item in nme_temp:
            nme_list.append(item)

    # nme
    print('nme: {:.4f}'.format(np.mean(nme_list)))
    # auc and failure rate
    failureThreshold = 0.1
    auc, failure_rate = compute_auc(nme_list, failureThreshold)
    print('auc @ {:.1f} failureThreshold: {:.4f}'.format(
        auc, failureThreshold ))
    print('failure_rate: {:}'.format(failure_rate))


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--result',
                        default="vamp_out",
                        type=str)
    parser.add_argument('--npz-txt',
                        default='npz_datalist.txt',
                        type=str)
    parser.add_argument('--gt',
                        default="../data/test_data/list.txt",
                        type=str)
    parser.add_argument('--show_image', action='store_true')
    parser.add_argument('--images', default='', type=str)
    parser.add_argument('--debug', default=False, type=str2bool)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)