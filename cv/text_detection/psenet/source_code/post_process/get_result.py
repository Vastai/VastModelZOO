# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import torch

from .pse import pse


def get_result(out,img_meta,rf,vis,bbox_type = 'rect'):
    outputs = dict()
    score = torch.sigmoid(out[:, 0, :, :])
    # kernel_num = 7
    kernels = out[:, :7, :, :] > 0
    text_mask = kernels[:, :1, :, :]
    kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
    score = score.data.cpu().numpy()[0].astype(np.float32)
    kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
    # min_area = 16
    label = pse(kernels, 16)

    # image size
    org_img_size = img_meta['org_img_size']
    img_size = img_meta['img_size']
    label_num = np.max(label) + 1
    label = cv2.resize(label, (img_size[1], img_size[0]),
                        interpolation=cv2.INTER_NEAREST)
    score = cv2.resize(score, (img_size[1], img_size[0]),
                        interpolation=cv2.INTER_NEAREST)
    scale = (float(org_img_size[1]) / float(img_size[1]),
            float(org_img_size[0]) / float(img_size[0]))
    bboxes = []
    scores = []
    for i in range(1, label_num):
        ind = label == i
        points = np.array(np.where(ind)).transpose((1, 0))

        if points.shape[0] < 16:
            label[ind] = 0
            continue

        score_i = np.mean(score[ind])
        # min_score=0.85
        if score_i < 0.85:
            label[ind] = 0
            continue

        # NOTE: cfg.test_cfg.bbox_type == 'rect':
        rect = cv2.minAreaRect(points[:, ::-1])
        bbox = cv2.boxPoints(rect) * scale

        if bbox_type == 'rect':
            rect = cv2.minAreaRect(points[:, ::-1])
            bbox = cv2.boxPoints(rect) * scale
        elif bbox_type == 'poly':
            binary = np.zeros(label.shape, dtype='uint8')
            binary[ind] = 1
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            bbox = contours[0] * scale

        bbox = bbox.astype('int32')
        bboxes.append(bbox.reshape(-1))
        scores.append(score_i)
    outputs.update(dict(bboxes=bboxes, scores=scores))

    # save result
    rf.write_result(img_meta, outputs)
    vis.process(img_meta, outputs)