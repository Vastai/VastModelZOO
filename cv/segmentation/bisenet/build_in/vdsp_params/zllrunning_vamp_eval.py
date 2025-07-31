
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
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = copy.deepcopy(im).astype(np.uint8)
    vis_parsing_anno = copy.deepcopy(parsing_anno).astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'_onehot.png', vis_parsing_anno)
        cv2.imwrite(save_path[:-4] +'_parsing.png', vis_parsing_anno_color)
        cv2.imwrite(save_path[:-4] +'_vis.png', vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_parsing_anno, vis_im


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="./dataset/sr/CelebAMask-HQ/bisegnet_test_img", help="hr gt image folder")
    parse.add_argument("--gt_dir", type=str, default="./dataset/sr/CelebAMask-HQ/bisegnet_test_mask", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="./code/vamc/vamp/0.2.0/npz_output", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[512, 512], help="vamp input shape")
    parse.add_argument("--draw_dir", type=str, default="npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=False, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()
    
    import sys
    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + '../../')
    from source_code.face_parsing.eval.metrics import SegMetric
    
    classes = 19
    metrics = SegMetric(n_classes=classes)
    metrics.reset()

    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".png")

            # gt image
            gt_image  = Image.open(os.path.join(args.src_dir, file_name.replace(".png", ".jpg")))
            resize_image = gt_image.resize(args.input_shape, Image.BILINEAR).convert('RGB')

            label = Image.open(os.path.join(args.gt_dir, file_name)).convert("L")
            label = label.resize(args.input_shape, Image.NEAREST)
            label = np.expand_dims(label, 0)

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            heatmap = np.load(output_npz_list[i], allow_pickle=True)["output_0"].astype(np.float32)
            vacc_pred = torch.from_numpy(heatmap)
            parsing = vacc_pred.squeeze(0).cpu().numpy().argmax(0)
            # draw
            vis_parsing_maps(resize_image, parsing, stride=1, save_im=True, save_path=os.path.join(args.draw_dir, file_name))
            
            # to eval
            pred = vacc_pred.data.max(1)[1].cpu().numpy()

            # eval metrics
            try:
                metrics.update(label, pred)
            except:
                continue

        score = metrics.get_scores()[0]
        class_iou = metrics.get_scores()[1]

        print("----------------- Total Performance --------------------")
        for k, v in score.items():
            print(k, v)

        print("----------------- Class IoU Performance ----------------")
        facial_names = ['background', 'skin', 'nose', 'eyeglass', 'left_eye', 'right_eye', 'left_brow', 'right_brow',
                        'left_ear', 'right_ear', 'mouth', 'upper_lip', 'lower_lip', 'hair', 'hat', 'earring', 'necklace',
                        'neck', 'cloth']
        for i in range(classes):
            print(facial_names[i] + "\t: {}".format(str(class_iou[i])))
        print("--------------------------------------------------------")
        ######################################################################################################


""" 
deploy_weights/bisenet-int8-kl_divergence-3_512_512-vacc
----------------- Total Performance --------------------
Overall Acc:     0.9561461254203113
Mean Acc :       0.8352879005724588
FreqW Acc :      0.9174539860097776
Mean IoU :       0.7445189693978729
Overall F1:      0.8423588941539634
----------------- Class IoU Performance ----------------
background      : 0.9384747596552021
skin    : 0.9286267585267922
nose    : 0.615762916637055
eyeglass        : 0.6082511419598616
left_eye        : 0.6511399782097205
right_eye       : 0.6433922711119691
left_brow       : 0.8349791326765159
right_brow      : 0.6726447217670521
left_ear        : 0.6572301933293974
right_ear       : 0.41897012526880467
mouth   : 0.8703918376949612
upper_lip       : 0.8361847016408601
lower_lip       : 0.7624986046966427
hair    : 0.7991596194205376
hat     : 0.8631332231598118
earring : 0.36317979831580316
necklace        : 0.8409039610289302
neck    : 0.9304766557663402
cloth   : 0.9104600176933283
--------------------------------------------------------
"""
