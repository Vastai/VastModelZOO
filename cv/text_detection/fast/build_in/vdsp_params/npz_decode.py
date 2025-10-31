# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import math
import cv2
import copy
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.nn import functional as F


def generate_bbox(keys, label, score, scales):
    label_num = len(keys)
    bboxes = []
    scores = []
    for index in range(1, label_num):
        i = keys[index]
        ind = (label == i)
        ind_np = ind.data.cpu().numpy()
        points = np.array(np.where(ind_np)).transpose((1, 0))
        if points.shape[0] < 200:
            label[ind] = 0
            continue
        score_i = score[ind].mean().item()
        if score_i < 0.88:
            label[ind] = 0
            continue

        # if cfg.test_cfg.bbox_type == 'rect':
        #     rect = cv2.minAreaRect(points[:, ::-1])
        #     alpha = math.sqrt(math.sqrt(points.shape[0] / (rect[1][0] * rect[1][1])))
        #     rect = (rect[0], (rect[1][0] * alpha, rect[1][1] * alpha), rect[2])
        #     bbox = cv2.boxPoints(rect) * scales

        # elif cfg.test_cfg.bbox_type == 'poly':
        binary = np.zeros(label.shape, dtype='uint8')
        binary[ind_np] = 1
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = contours[0] * scales
        bbox = bbox.astype('int32')
        bboxes.append(bbox.reshape(-1).tolist())
        scores.append(score_i)
    return bboxes, scores

def get_results(out, ori_h, ori_w, scale=2):

    out = torch.Tensor(out)

    pooling_2s = nn.MaxPool2d(kernel_size=9//2+1, stride=1, padding=(9//2) // 2)

    org_img_size = [ori_h, ori_w]
    img_size = [512, 512]  # 640*640
    batch_size = out.size(0)

    texts = F.interpolate(out[:, 0:1, :, :], size=(img_size[0] // scale, img_size[1] // scale),
                            mode='nearest')  # B*1*320*320
    texts = pooling_2s(texts)  # B*1*320*320
    score_maps = torch.sigmoid_(texts)  # B*1*320*320
    score_maps = F.interpolate(score_maps, size=(img_size[0], img_size[1]), mode='nearest')  # B*1*640*640
    score_maps = score_maps.squeeze(1)  # B*640*640
    
    kernels = (out[:, 0, :, :] > 0).to(torch.uint8)  # B*160*160
    labels_ = []
    for kernel in kernels.numpy():
        ret, label_ = cv2.connectedComponents(kernel)
        labels_.append(label_)
    labels_ = np.array(labels_)
    labels_ = torch.from_numpy(labels_)
    labels = labels_.unsqueeze(1).to(torch.float32)  # B*1*160*160
    labels = F.interpolate(labels, size=(img_size[0] // scale, img_size[1] // scale), mode='nearest')  # B*1*320*320
    labels = pooling_2s(labels)
    labels = F.interpolate(labels, size=(img_size[0], img_size[1]), mode='nearest')  # B*1*640*640
    labels = labels.squeeze(1).to(torch.int32)  # B*640*640

    keys = [torch.unique(labels_[i], sorted=True) for i in range(batch_size)]


    scales = (float(org_img_size[1]) / float(img_size[1]),
                float(org_img_size[0]) / float(img_size[0]))

    results = []
    for i in range(batch_size):
        bboxes, scores = generate_bbox(keys[i], labels[i], score_maps[i], scales)
        results.append(dict(
            bboxes=bboxes,
            scores=scores
        ))

    return results


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="./datasets/ocr/ctw1500/test_images/", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="./datasets/ocr/ctw1500/npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="../../source_code/npz_output/", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[512, 512], help="vamp input shape")

    args = parse.parse_args()
    print(args)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()
    
    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".png")

            # src image
            ori_image  = cv2.imread(os.path.join(args.src_dir, file_name.replace(".png", ".jpg")))

            # load npy
            npz_file = output_npz_list[i]
            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)
            
            result = get_results(heatmap, ori_image.shape[0], ori_image.shape[1])
            # print(results)
            f = open('./vsx_int8_pred/' + file_name.split('.')[0] + '.txt', 'w')
            for i in range(len(result)):
                box = result[i]['bboxes']
                for b in box:
                    b = [str(bb) for bb in b]
                    f.writelines(' '.join(b) + '\n')
            
            f.close()

            
