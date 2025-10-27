# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn


class Bicubic_plus_plus(nn.Module):
    def __init__(self, sr_rate=3):
        super(Bicubic_plus_plus, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv_out = nn.Conv2d(32, (2*sr_rate)**2 * 3, kernel_size=3, padding=1, bias=False)
        self.Depth2Space = nn.PixelShuffle(2*sr_rate)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.1)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.act(x0)
        x1 = self.conv1(x0)
        x1 = self.act(x1)
        x2 = self.conv2(x1)
        x2 = self.act(x2) + x0
        y = self.conv_out(x2)
        y = self.Depth2Space(y)
        return y


if __name__ == '__main__':
    import torch
    device = 'cpu'
    model = Bicubic_plus_plus(sr_rate=2).to(device)
    # model = Bicubic_plus_plus(sr_rate=3).to(device)

    checkpoint = "ckpts/Bicubic_plus_plus_epoch=3959_val_psnr=32.46.pth"
    state_dict = torch.load(checkpoint, map_location="cpu")['state_dict']
    state_dict = {k.replace("network.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 1080, 1920).to('cpu')
    flops, params = profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)

    # pretrained/bicubic_pp_x3.pth 1080_1920 3x
    # flops(G): 58.061
    # params: 50.400K

    # ckpts/Bicubic_plus_plus_epoch=999_val_psnr=32.36.ckpt 1080_1920 2x
    # flops(G): 38.154
    # params: 33.120K

    # ckpts4/Bicubic_plus_plus_epoch=1819_val_psnr=24.30.ckpt 1080_1920 4x
    # flops(G): 85.930
    # params: 74.592K

    input_shape = (1, 3, 1080, 1920) # nchw
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, input_data)#.eval()
        scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))
        # scripted_model = torch.jit.load(checkpoint.replace(".pth", ".torchscript.pt"))

    # onnx==10.0.0，opset 10 
    # opset_version=10 can not export onnx in pixel_shuffle
    import onnx
    with torch.no_grad():
        torch.onnx.export(model, input_data, checkpoint.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
        # dynamic_axes= {
        #             "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
        #             "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
        )
    ###############################################################################