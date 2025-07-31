# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import torch
import torch.nn as nn
import numpy as np
import torchvision


def export_model(checkpoint):
    model = torchvision.models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 18)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device)
    model.eval()

    ###############################################################################

    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)

    # res34_fair_align_multi_7_20190809.pt 224 224
    # flops(G): 8.157
    # params: 21.294M

    input_shape = (1, 3, 224, 224) # nchw
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, input_data)

    # onnx==10.0.0，opset 10 
    # import onnx
    with torch.no_grad():
        torch.onnx.export(model, input_data, checkpoint.replace(".pt", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
        # dynamic_axes= {
        #             "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
        #             "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
        )

    ################################################################################
    return model


if __name__ == "__main__":    
    device = torch.device("cpu")
    model = export_model("res34_fair_align_multi_7_20190809.pt")
