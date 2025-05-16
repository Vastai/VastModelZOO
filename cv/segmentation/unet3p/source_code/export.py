
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :    zpwen
@Email  :    algorithm@vastaitech.com
@Time   :    2025/04/24 16:17:38
'''

import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F

from unet import UNet3Plus, UNet3Plus_DeepSup



if __name__ == '__main__':

    net = UNet3Plus(n_channels=3, n_classes=1)
    # net = UNet3Plus_DeepSup(n_channels=3, n_classes=1)

    checkpoint = "unet3p.pth"
    # checkpoint = "unet3p_deepsupervision.pth"

    device = torch.device('cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(checkpoint, map_location=device))

    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 128, 128)
    flops, params = profile(net, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)
    
    # unet3p.pth (1, 3, 128, 128)
    # flops(G): 110.928
    # params: 26.972M

    # unet3p_deepsupervision.pth (1, 3, 128, 128)
    # flops(G): 110.965
    # params: 26.990M

    ###########################################################################################
    input_shape = [1, 3, 128, 128]
    input_data = torch.randn(input_shape)
    with torch.no_grad():
        scripted_model = torch.jit.trace(net.eval(), input_data, strict=False).eval()
        scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))
        torch.onnx.export(net.eval(), input_data, checkpoint.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11)
    
    # import onnx, onnxruntime, cv2
    # session = onnxruntime.InferenceSession(checkpoint.replace(".pth", ".onnx"))
    # image_src = cv2.imread(args.input[0]).astype('float32')
    # resized = cv2.resize(image_src, (128, 128), interpolation=cv2.INTER_LINEAR)
    # img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # img_in /= 255.0
    # img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    # img_in = np.expand_dims(img_in, axis=0)

    # input_name = session.get_inputs()[0].name
    # pred = session.run(None, {input_name: img_in})[0]
    # print(pred)
    ######################onnx####################################
