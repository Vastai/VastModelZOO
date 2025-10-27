# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import json
import os

import onnx
import torch
from mmcv import Config
from onnxsim import simplify

from models import build_model
from models.utils import fuse_module


def summary_helper(module):
    detail_str = ""
    sub_modules = len(list(module.children()))
    if  sub_modules == 0:
        # class_name = str(module.__class__).split(".")[-1].split("'")[0]
        oprator_details = module.__str__()
        detail_str = "{} \t|\t 1\n".format(oprator_details)
    else:
        for m in module.children():
            detail_str += summary_helper(m)

    return detail_str


def summary(model, input_size=None, batch_size=-1, device=None, dtypes=None):
    detail_str = summary_helper(model)
    print(detail_str)
    return None, detail_str



def main(args):
    input_shape = [1, 3, 1024, 1760]
    output_path = args.checkpoint.split('.pth')[0] + "-" + str(int(input_shape[1])) + "_" + str(int(input_shape[2])) + "_" + str(int(input_shape[3])) + ".onnx"

    data ={}
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(report_speed=args.report_speed))
    print(json.dumps(cfg._cfg_dict, indent=4))
    model = build_model(cfg.model)
    model = model
    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(
                args.checkpoint))

            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            raise

    model = fuse_module(model)
    model.eval()
    if args.op:
        _, detail_str = summary(model)
        with open('./weights/psenet.txt', "w") as fw:
            fw.write(detail_str)
    # device = torch.device("cuda:0")
    x = torch.randn(input_shape)#.to(device)
    data['imgs'] = x
    data.update(dict(cfg=cfg))
    with torch.no_grad():
        if args.dynamic:
            torch.onnx.export(model, x, output_path, input_names=["input"], opset_version=10,dynamic_axes = {'input':[ 2, 3]})
        else:
            torch.onnx.export(model, x, output_path, input_names=["input"], opset_version=10)


    onnx_model = onnx.load(output_path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)
    print('finished exporting onnx')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', action='store_true')
    parser.add_argument('--op', action='store_true')
    parser.add_argument('--dynamic', action='store_true')
    args = parser.parse_args()

    main(args)
