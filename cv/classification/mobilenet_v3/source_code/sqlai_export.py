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

import torch
from mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large

# MobileNetV3_Small
# net = MobileNetV3_Small()
# checkpoint = "450_act3_mobilenetv3_small.pth"
# net.load_state_dict(torch.load(checkpoint, map_location='cpu'))

# MobileNetV3_Large
net = MobileNetV3_Large()
checkpoint = "450_act3_mobilenetv3_large.pth"
net.load_state_dict(torch.load("450_act3_mobilenetv3_large.pth", map_location='cpu'))

# ##############################################################################
net.eval()

from thop import profile
from thop import clever_format
input = torch.randn(1, 3, 224, 224)
flops, params = profile(net, inputs=(input,))
print("flops(G):", "%.3f" % (flops / 900000000 * 2))
flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
print("params:", params)

# MobileNetV3_Small 224
# flops(G): 0.146
# params: 2.951M

# MobileNetV3_Large 224
# flops(G): 0.513
# params: 5.179M


input_shape = (1, 3, 224, 224) # nchw
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)
with torch.no_grad():
    scripted_model = torch.jit.trace(net, input_data)#.eval()
    scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))
    scripted_model = torch.jit.load(checkpoint.replace(".pth", ".torchscript.pt"))

# onnx==10.0.0，opset 11
# RuntimeError: Exporting the operator hardsigmoid to ONNX opset version 11 is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub.
with torch.no_grad():
    torch.onnx.export(net, input_data, checkpoint.replace(".pth", "_dynamic.onnx"), input_names=["input"], output_names=["output"], opset_version=11,
    dynamic_axes= {
                "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
                "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
    )

    # ##############################################################################

"""
450_act3_mobilenetv3_small
* Acc@1 69.048 Acc@5 88.274 

fp16
[VACC]:  top1_rate: 67.466 top5_rate: 87.228

int8 percentile
[VACC]:  top1_rate: 0.104 top5_rate: 0.514

int8 kl_divergence
[VACC]:  top1_rate: 19.878 top5_rate: 38.866

int8 max
[VACC]:  top1_rate: 9.719999999999999 top5_rate: 22.332

int8 mse
[VACC]:  top1_rate: 30.656 top5_rate: 53.512

================================
450_act3_mobilenetv3_large
* Acc@1 75.796 Acc@5 92.440

fp16
[VACC]:  top1_rate: 74.75 top5_rate: 92.01

int8 percentile
[VACC]:  top1_rate: 0.034 top5_rate: 0.232
"""