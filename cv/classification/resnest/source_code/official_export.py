# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :    tonyx
@Email  :    algorithm@vastaitech.com
@Time   :    2025/04/27 10:17:38
'''

import argparse
import os

import thop
import torch
from thop import clever_format

parse = argparse.ArgumentParser(description="MAKE MODELS CLS FOR VACC")
parse.add_argument("--model_name", type=str, default="resnest50")
parse.add_argument("--save_dir", type=str, default="./")
parse.add_argument("--size", type=int, default=224)
args = parse.parse_args()
print(args)


def count_op(model, input):
    flops, params = thop.profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)

def export(opt):
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    img = torch.randn(1, 3, opt.size, opt.size)
    # get list of models
    torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
    # load pretrained models, using ResNeSt-50 as an example
    net = torch.hub.load('zhanghang1989/ResNeSt', opt.model_name, pretrained=True)
    torch.onnx.export(net, img, os.path.join(opt.save_dir,opt.model_name + ".onnx"), input_names=["input"], opset_version=11,dynamic_axes = {'input':[ 2, 3]})
    print("convert ok!")
    count_op(net, img)


if __name__ == "__main__":
    export(args)