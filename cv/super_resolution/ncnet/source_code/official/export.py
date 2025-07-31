# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch
import numpy as np
import onnx
from ncnet import NCNet

if __name__ == '__main__':
    checkpoint = "RCAN_TrainCodeN/experiment/NCNET_BIX2_G10R20P482/model/model_best.pt"

    device = 'cpu'
    model = NCNet().to(device)
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # ##############################################################################

    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 1080, 1920).to(device)
    flops, params = profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)

    # RCAN_MODIFY 1080_1920
    # flops(G): 358.207
    # params: 297.723K

    # RCAN_TrainCodeN/experiment/NCNET_BIX2_G10R20P482/model/model_best.pt  1080_1920
    # flops(G): 195.748
    # params: 42.664K

    input_shape = (1, 3, 1080, 1920) # nchw
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, input_data)#.eval()
        scripted_model.save(checkpoint.replace(".pt", ".torchscript.pt"))
        scripted_model = torch.jit.load(checkpoint.replace(".pt", ".torchscript.pt"))

    # onnx==10.0.0，opset 10 
    # opset_version=10 can not export onnx in pixel_shuffle
    with torch.no_grad():
        torch.onnx.export(model, input_data, checkpoint.replace(".pt", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
        # dynamic_axes= {
        #             "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
        #             "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
        )

        # ##############################################################################