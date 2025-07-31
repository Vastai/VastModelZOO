# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import torch

from edsr_arch_modify import EDSR

checkpoint = "ckpt_gopro.pth"

model = EDSR(3, 3, 32, 9, 1)
model_weights = torch.load(checkpoint, map_location=torch.device('cpu'))
model.load_state_dict(model_weights)
model.eval()


# ##############################################################################

# from thop import profile
# from thop import clever_format
# input = torch.randn(1, 3, 1280, 720)
# flops, params = profile(model, inputs=(input,))
# print("flops(G):", "%.3f" % (flops / 900000000 * 2))
# flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
# print("params:", params)

# edsr 1x 1280_720
# flops(G): 363.469
# params: 177.475K

input_shape = (1, 3, 1280, 720) # nchw
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)
with torch.no_grad():
    scripted_model = torch.jit.trace(model, input_data)#.eval()
    scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))
    scripted_model = torch.jit.load(checkpoint.replace(".pth", ".torchscript.pt"))

# onnx==10.0.0，opset 13 
# opset 10 can not export onnx in pixel_shuffle
# import onnx
# with torch.no_grad():
#     torch.onnx.export(model, input_data, checkpoint.replace(".pth", "_dynamic.onnx"), input_names=["input"], output_names=["output"], opset_version=13,
#     dynamic_axes= {
#                 "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
#                 "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
#     )

    # ##############################################################################
