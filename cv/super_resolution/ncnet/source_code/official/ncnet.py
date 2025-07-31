# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def make_model(args, parent=False):
    return NCNet()

class UpOnly(nn.Sequential):
    def __init__(self, scale):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError

        super(UpOnly, self).__init__(*m)


class NCNet(nn.Module):
    def __init__(self, n_feats=32, out_c=3, scale_factor=2):
        super(NCNet, self).__init__()

        ps_feat = out_c*(scale_factor**2)

        self.nearest_weight = torch.eye(out_c).repeat(1, scale_factor**2).reshape(ps_feat, out_c).to(torch.device('cuda'))
        self.nearest_weight = self.nearest_weight.unsqueeze(-1).unsqueeze(-1)
        
        # define body module
        self.body = nn.Sequential(
            nn.Conv2d(out_c, n_feats, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(n_feats, ps_feat, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(ps_feat, ps_feat, 3, 1, 1))
        
        self.upsample = UpOnly(scale_factor)

    def forward(self, x):
        x_res = F.conv2d(x, self.nearest_weight)
        x = self.body(x)
        x += x_res
        x = self.upsample(x)
        return x



if __name__ == '__main__':
    device = 'cpu'
    model = NCNet().to(device)
    checkpoint = "RCAN_TrainCodeN/experiment/NCNET_BIX2_G10R20P482/model/model_best.pt"
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
    import onnx
    with torch.no_grad():
        torch.onnx.export(model, input_data, checkpoint.replace(".pt", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
        # dynamic_axes= {
        #             "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
        #             "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
        )

        # ##############################################################################