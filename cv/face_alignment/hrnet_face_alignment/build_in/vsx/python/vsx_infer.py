# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import cv2
import torch
import glob
import argparse
import numpy as np
# from tqdm.contrib import tzip
from tqdm import tqdm
from typing import  List, Union
from scipy.integrate import simps
import vaststreamx as vsx
from matplotlib import pyplot as plt

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

def cv_rgb_image_to_vastai(image_cv, device_id):
    assert len(image_cv.shape) >= 2
    h = image_cv.shape[0]
    w = image_cv.shape[1]
    if len(image_cv.shape) == 3:
        return vsx.create_image(
            image_cv, vsx.ImageFormat.BGR_INTERLEAVE, w, h, device_id
        )
    elif len(image_cv.shape) == 2:
        return vsx.create_image(image_cv, vsx.ImageFormat.GRAY, w, h, device_id)
    else:
        raise Exception("unsupported ndarray shape", image_cv.shape)

class ModelBase:
    def __init__(
        self, model_prefix, vdsp_config, batch_size=1, device_id=0, hw_config=""
    ) -> None:
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.preproc_ops_ = vsx.Operator.load_ops_from_json_file(vdsp_config)
        assert len(self.preproc_ops_) > 0, "load_ops_from_json_file failed"
        self.graph_ = vsx.Graph()
        self.graph_.add_operators(*self.preproc_ops_, self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

    @property
    def batch_size(self):
        return self.model_.batch_size

    @property
    def max_batch_size(self):
        return self.model_.max_batch_size

    @property
    def input_count(self):
        return self.model_.input_count

    @property
    def output_count(self):
        return self.model_.output_count

    @property
    def input_shape(self):
        return self.model_.input_shape

    @property
    def output_shape(self):
        return self.model_.output_shape

class ModelCV(ModelBase):
    def __init__(
        self, model_prefix, vdsp_config, batch_size=1, device_id=0, hw_config=""
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)

    def process(
        self, input: Union[List[np.ndarray], List[vsx.Image], np.ndarray, vsx.Image]
    ):
        if isinstance(input, list):
            if isinstance(input[0], np.ndarray):
                return self.process(
                    [cv_rgb_image_to_vastai(x, self.device_id_) for x in input]
                )
            else:
                return self.process_impl(input)
        return self.process([input])[0]

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height = input_shape[-2]
        width = input_shape[-1]
        dummy = np.zeros((height, width, 3), dtype=dtype)
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            vacc_dummy = vsx.create_image(
                dummy, vsx.ImageFormat.BGR_INTERLEAVE, width, height, self.device_id_
            )
            return [vacc_dummy] * batch_size

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [[vsx.as_numpy(o) for o in out] for out in outputs]
        

def parser_args():
    parse = argparse.ArgumentParser(description="RUN WITH VSX")
    parse.add_argument(
        "--data_dir",
        type=str,
        default="imgs"
    )
    parse.add_argument(
        "--model_prefix_path",
        type=str,
        default="/path/to/model/mod",
        help="model info")
    parse.add_argument(
        "--vdsp_params_info",
        type=str,
        default="../build_in/vdsp_params/pytorch-hrnetv2-vdsp_params.json",
        help="vdsp op info",
    )
    parse.add_argument("--npz_datalist", type = str, default = "./npz_datalist.txt", help = "npz dir path")
    parse.add_argument("--device_id", type=int, default=0, help="device id")
    parse.add_argument("--batch", type=int, default=1, help="bacth size")
    parse.add_argument("--save_dir", type=str, default="./vsx_results", help="save result")
    args = parse.parse_args()

    return args


if __name__ == '__main__':
    args = parser_args()
    os.makedirs(args.save_dir, exist_ok=True)
    npz_filt_list = []
    with open(args.npz_datalist, 'r') as fr:
        for line in fr:
            npz_filt_list.append(line.strip())

    file_list = glob.glob(os.path.join(args.data_dir, "*", "*.jpg"))
    print(len(file_list))
    print(len(npz_filt_list))
    face_aligner = ModelCV(args.model_prefix_path, args.vdsp_params_info, args.batch, args.device_id)
    for file in tqdm(file_list):
        file = file.strip()
        filename = os.path.basename(file)
        image = cv2.imread(file)
        assert image is not None, f"Read image failed:{file}"
        result = face_aligner.process(image)
        landmarks = result[0]
        out = {"output_0": landmarks}
        image_name = os.path.basename(file).split('.')[0]
        # 需要从npz_filt_list中去取文件名才能对齐
        # 多的图片不做写入
        for npz_file in npz_filt_list:
            if image_name in npz_file:
                npz_file = npz_file.strip()
                image_name = npz_file
                print(image_name)
                np.savez(os.path.join(args.save_dir, image_name), **out)

