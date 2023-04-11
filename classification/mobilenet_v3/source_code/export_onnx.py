# ==============================================================================
# 
# Copyright (C) 2023 VastaiTech Technologies Inc.  All rights reserved.
# 
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :     vastml
@Time  : 2023/04/07 22:08:58
'''

from mobileNetV3 import MobileNetV3
from collections import OrderedDict
import torch
import thop
from thop import clever_format

def count_op(model, input):
    flops, params = thop.profile(model, inputs=(input,))
    print("flops(G):", "%.5f" % (flops / 1000**3))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)
model = MobileNetV3(mode="small", classes_num=1000, input_size=224, 
                    width_multiplier=1, dropout=0.2, 
                    BN_momentum=0.1, zero_gamma=True)
new_state_dict = OrderedDict()
model.load_state_dict(torch.load("./pretrained/best_model_wts-67.52.pth"))
model.eval()
img = torch.randn(1,3,224,224)
count_op(model,img)
torch.onnx.export(model,img,"./mobilenetv3-small.onnx", input_names=["input"], opset_version=10)
print("export onnx successfully!")