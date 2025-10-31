# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
#device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
device = torch.device('cpu')

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path,map_location="cpu"), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))


###########################################################################
from thop import profile
from thop import clever_format
input = torch.randn(1, 3, 128, 128)
flops, params = profile(model, inputs=(input,))
print("flops(G):", "%.3f" % (flops / 900000000 * 2))
flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
print("params:", params)

# RRDB_ESRGAN_x4 128
# flops(G): 653.675
# params: 16.698M

checkpoint = model_path
sr_model = model
input_shape = (1, 3, 128, 128)
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)

scripted_model = torch.jit.trace(sr_model, input_data)#.eval()
scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))
scripted_model = torch.jit.load(checkpoint.replace(".pth", ".torchscript.pt"))

# onnx==10.0.0，opset 10
import onnx
torch.onnx.export(sr_model, input_data, checkpoint.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=10)
shape_dict = {"input": input_shape}
onnx_model = onnx.load(checkpoint.replace(".pth", ".onnx"))
###########################################################################


idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
