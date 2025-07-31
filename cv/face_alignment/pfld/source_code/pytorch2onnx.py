# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import argparse
from models.pfld import PFLDInference
from torch.autograd import Variable
import torch
import onnxsim


def count_op(model, input):
    from thop import profile, clever_format
    flops, params = profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)



parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--torch_model',
                    default="./checkpoint/snapshot/checkpoint.pth.tar")
parser.add_argument('--onnx_model', default="./output/pfld.onnx")
parser.add_argument('--onnx_model_sim',
                    help='Output ONNX model',
                    default="./output/pfld-sim.onnx")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
checkpoint = torch.load(args.torch_model, map_location=torch.device('cpu'))
pfld_backbone = PFLDInference()
pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
print("PFLD bachbone:", pfld_backbone)

print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(1, 3, 112, 112))
input_names = ["input_1"]
output_names = ["output_1"]
count_op(pfld_backbone, dummy_input)
torch.onnx.export(pfld_backbone,
                dummy_input,
                args.onnx_model,
                verbose=True,
                input_names=input_names,
                output_names=output_names)

print("====> check onnx model...")
import onnx
model = onnx.load(args.onnx_model)
onnx.checker.check_model(model)

print("====> Simplifying...")
model_opt, check = onnxsim.simplify(args.onnx_model)
# model_opt = onnxsim.simplify(args.onnx_model)
# print("model_opt", model_opt)
onnx.save(model_opt, args.onnx_model_sim)
print("onnx model simplify Ok!")
