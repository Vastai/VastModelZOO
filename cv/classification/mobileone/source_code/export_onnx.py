# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import os
import argparse
from mobileone import mobileone
from thop import profile, clever_format

parse = argparse.ArgumentParser(description="IMAGENET TOPK")
parse.add_argument("--input", default="../pretrained/mobileone_s4.pth.tar", type=str)
args = parse.parse_args()

def count_op(model, input):
    flops, params = profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)
var = os.path.basename(args.input).split('_')[-1].split('.')[0]
model = mobileone(variant=var, inference_mode=True)
checkpoint = torch.load(args.input, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
img = torch.randn(1,3,224,224)
count_op(model,img)
save_path = "../onnx/"+os.path.basename(args.input).split('.')[0]+".onnx"
torch.onnx.export(model, img, save_path, input_names=["input"], opset_version=10)
print("export done")