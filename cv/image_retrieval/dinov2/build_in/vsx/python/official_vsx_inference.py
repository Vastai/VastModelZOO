# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import cv2
import argparse
import torch
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
from threading import Thread, Event
from typing import Dict, Generator, Iterable, List, Union

import vaststreamx as vsx
from normalize_op import NormalizeOp, NormalType
from space_to_depth_op import SpaceToDepthOp

import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../../../source_code')
import utils


def cv_rgb_image_to_vastai(image_cv, device_id):
    assert len(image_cv.shape) >= 2
    h, w = image_cv.shape[:2]
    if len(image_cv.shape) == 3:
        return vsx.create_image(
            image_cv, vsx.ImageFormat.BGR_INTERLEAVE, w, h, device_id
        )
    elif len(image_cv.shape) == 2:
        return vsx.create_image(image_cv, vsx.ImageFormat.GRAY, w, h, device_id)
    else:
        raise Exception("unsupported ndarray shape", image_cv.shape)


class Dinov2Model:
    def __init__(
        self,
        model_prefix,
        norm_op_elf,
        space2depth_op_elf,
        batch_size=1,
        device_id=0,
        hw_config="",
        do_copy=False,
    ) -> None:
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0

        # normalize op
        mean = np.array([14275, 14156, 13951], dtype=np.uint16)
        std = np.array([13140, 13099, 13107], dtype=np.uint16)
        norm_type = NormalType.NORMAL_DIV255_MINUSMEAN_DIVSTD
        self.normalize_op_ = NormalizeOp(
            elf_file=norm_op_elf,
            device_id=device_id,
            mean=mean,
            std=std,
            norm_type=norm_type,
        )

        # space_to_depth op
        kh, kw, out_h, out_w = 14, 14, 256, 784
        self.space_to_depth_op_ = SpaceToDepthOp(
            kh=kh,
            kw=kw,
            oh_align=out_h,
            ow_align=out_w,
            elf_file=space2depth_op_elf,
            device_id=device_id,
        )

        # model
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.graph_ = vsx.Graph(do_copy)
        self.graph_.add_operators(self.model_op_)
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

    def process_impl(self, inputs):
        mod_h, mod_w = self.model_.input_shape[0][-2:]
        outputs = []
        np.set_printoptions(threshold=np.inf)
        infer_input = []
        for input in inputs:
            vacc_inputs = []
            cvtcolor_out = vsx.cvtcolor(input, vsx.ImageFormat.RGB_PLANAR)
            resize_out = vsx.resize(
                cvtcolor_out,
                vsx.ImageResizeType.BILINEAR_PILLOW,
                resize_width=mod_w,
                resize_height=mod_h,
            )
            
            norm_out = self.normalize_op_.process(resize_out)
            space_to_depth_out = self.space_to_depth_op_.process(norm_out)
            vacc_inputs.append(space_to_depth_out)
            infer_input.append(vacc_inputs)
        model_outs = self.stream_.run_sync(infer_input)
        for model_out in model_outs:
            outputs.append(vsx.as_numpy(model_out[0]).astype(np.float32))

        return outputs


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="deploy_weights/official_dinov2_fp16/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--norm_elf_file",
        default="/path/to/elf/normalize",
        help="normalize op elf file",
    )
    parser.add_argument(
        "--space_to_depth_elf_file",
        default="/path/to/elf/space_to_depth",
        help="space_to_depth op elf files",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="batch_size to run",
    )
    parser.add_argument(
        "--input_file",
        default="/path/to/jpg/worcester_000198.jpg",
        help="input file",
    )

    parser.add_argument(
        "--dataset_root",
        default="/path/to/roxford5k/jpg",
        help="input dataset root",
    )
    parser.add_argument(
        "--dataset_conf",
        default="/path/to/roxford5k/gnd_roxford5k.pkl",
        help="dataset conf pkl file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()

    dinov2 = Dinov2Model(
        args.model_prefix,
        args.norm_elf_file,
        args.space_to_depth_elf_file,
        batch_size=args.batch_size,
        device_id=args.device_id,
    )

    if args.dataset_root == "":
        image = cv2.imread(args.input_file)
        assert image is not None, f"Failed to read input file: {args.input_file}"
        outputs = dinov2.process(image)     
        for output in outputs:
            print(f"output:{output} ")
    else:
        train_list = []
        query_list = []
        with open(args.dataset_conf, 'rb') as f:
            cfg = pickle.load(f)
        query_list = cfg["qimlist"]
        train_list = cfg["imlist"]
        
        train_features = []
        query_features = []
        for file in tqdm(train_list):
            fullname = os.path.join(args.dataset_root, file+".jpg")
            image = cv2.imread(fullname)
            assert image is not None, f"Failed to read input file: {fullname}"
            outputs = dinov2.process(image)
            train_features.append(torch.from_numpy(outputs[0]))
        for file in tqdm(query_list):
            fullname = os.path.join(args.dataset_root, file+".jpg")
            image = cv2.imread(fullname)
            assert image is not None, f"Failed to read input file: {fullname}"
            outputs = dinov2.process(image)
            query_features.append(torch.from_numpy(outputs[0]))
        # normalize features
        train_features = torch.stack(train_features, dim=1)
        train_features = torch.squeeze(train_features)
        train_features = nn.functional.normalize(train_features.to(torch.float32), dim=1, p=2)
        
        query_features = torch.stack(query_features, dim=1)
        query_features = torch.squeeze(query_features)
        query_features = nn.functional.normalize(query_features.to(torch.float32), dim=1, p=2)
        # query_features = query_features.reshape(70,1024)
        # Step 2: similarity
        sim = torch.mm(train_features.T, query_features)
        ranks = torch.argsort(-sim, dim=0).cpu().numpy()
        
        # Step 3: evaluate
        gnd = cfg['gnd']
        # evaluate ranks
        ks = [1, 5, 10]
        # search for easy & hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = utils.compute_map(ranks, gnd_t, ks)
        # search for hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = utils.compute_map(ranks, gnd_t, ks)
        print('>> {}: mAP M: {}, H: {}'.format(args.dataset_root, np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        print('>> {}: mP@k{} M: {}, H: {}'.format(args.dataset_root, np.array(ks), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))


'''
torch
dinov2_vitl14: dinov2_vitl14_pretrain.pth
>> roxford5k: mAP M: 74.91, H: 54.14
>> roxford5k: mP@k[ 1  5 10] M: [95.71 95.14 90.57], H: [91.43 73.71 63.11]


dinov2_vitl14_reg: dinov2_vitl14_reg4_pretrain.pth 
>> roxford5k: mAP M: 79.13, H: 59.21
>> roxford5k: mP@k[ 1  5 10] M: [97.14 95.14 91.57], H: [94.29 79.71 69.7 ]
'''

'''
onnx
dinov2_vitl14_reg: dinov2_vitl14_reg4.onnx 送测模型
>> roxford5k: mAP M: 79.41, H: 59.48
>> roxford5k: mP@k[ 1  5 10] M: [97.14 95.71 92.  ], H: [95.71 79.71 69.86]
'''

'''
vacc
dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod
>> ./datasets/image_retrieval/roxford5k/jpg: mAP M: 79.6, H: 58.18
>> ./datasets/image_retrieval/roxford5k/jpg: mP@k[ 1  5 10] M: [98.57 94.52 91.38], H: [92.86 80.05 70.05]
'''