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

# from basicsr.utils.registry import ARCH_REGISTRY

import torch.nn as nn
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=False):
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):

    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):

    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class ESA(nn.Module):

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class RRFB(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16):
        super(RRFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out_33 = (self.c1_r(x))
        out = self.act(out_33)

        out_33 = (self.c2_r(out))
        out = self.act(out_33)

        out_33 = (self.c3_r(out))
        out = self.act(out_33)

        out = out + x
        out = self.esa(self.c5(out))

        return out


class RRFB_NoSE(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None):
        super(RRFB_NoSE, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out_33 = (self.c1_r(x))
        out = self.act(out_33)

        out_33 = (self.c2_r(out))
        out = self.act(out_33)

        out_33 = (self.c3_r(out))
        out = self.act(out_33)

        out = out + x
        out = self.c5(out)

        return out

def make_model(args, parent=False):
    model = DIPNet()
    return model


# @ARCH_REGISTRY.register()
class DIPNet(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=44,
                 mid_channels=38,
                 upscale=2):
        super(DIPNet, self).__init__()

        self.conv_1 = conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.block_1 = RRFB(feature_channels, mid_channels)
        self.block_2 = RRFB(feature_channels, mid_channels)
        self.block_3 = RRFB(feature_channels, mid_channels)
        self.block_4 = RRFB(feature_channels, mid_channels)

        self.conv_2 = conv_layer(feature_channels,
                                 feature_channels,
                                 kernel_size=3)

        self.upsampler = pixelshuffle_block(feature_channels,
                                            out_channels,
                                            upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)


        out_low_resolution = self.conv_2(out_b4) + out_feature
        output = self.upsampler(out_low_resolution)

        return output


# @ARCH_REGISTRY.register()
class DIPNet_NoSE(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=44,
                 mid_channels=38,
                 upscale=2):
        super(DIPNet_NoSE, self).__init__()

        self.conv_1 = conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.block_1 = RRFB_NoSE(feature_channels, mid_channels)
        self.block_2 = RRFB_NoSE(feature_channels, mid_channels)
        self.block_3 = RRFB_NoSE(feature_channels, mid_channels)
        self.block_4 = RRFB_NoSE(feature_channels, mid_channels)

        self.conv_2 = conv_layer(feature_channels,
                                 feature_channels,
                                 kernel_size=3)

        self.upsampler = pixelshuffle_block(feature_channels,
                                            out_channels,
                                            upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)


        out_low_resolution = self.conv_2(out_b4) + out_feature
        output = self.upsampler(out_low_resolution)

        return output


if __name__ == "__main__":

    # model = DIPNet(
    #     in_channels=3,
    #     out_channels=3,
    #     feature_channels=44,
    #     mid_channels=38,
    #     upscale=2)

    # model_path = "experiments/100_train_DIPNet_x2_m4c16_prelu_RGB/models/net_g_137600.pth"

    # model = DIPNet(
    #     in_channels=3,
    #     out_channels=3,
    #     feature_channels=38,
    #     mid_channels=32,
    #     upscale=2)

    # model_path = "experiments/100_train_DIPNet_x2_m4c16_prelu_RGB_mini/models/net_g_108800.pth"

    # model = DIPNet_NoSE(
    #     in_channels=3,
    #     out_channels=3,
    #     feature_channels=44,
    #     mid_channels=38,
    #     upscale=2)


    # model_path = "experiments/100_train_DIPNet_x2_m4c16_prelu_RGB_noSE/models/net_g_227200.pth"

    model = DIPNet_NoSE(
        in_channels=3,
        out_channels=3,
        feature_channels=32,
        mid_channels=24,
        upscale=2)
    model_path = "dipnet_2x.pth"

    # model = DIPNet_NoSE(
    #     in_channels=3,
    #     out_channels=3,
    #     feature_channels=16,
    #     mid_channels=12,
    #     upscale=2)
    # model_path = "experiments/100_train_DIPNet_x2_m4c16_prelu_RGB_noSE_mini_small/models/net_g_1596800.pth"


    device = torch.device("cpu")

    state_dict = torch.load(model_path)['params']
    model.load_state_dict(state_dict, strict=True)

    # print(model)
    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # ##############################################################################

    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 1080, 1920)
    flops, params = profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)

    # model_zoo/DIPNet.pth 640, 960 4x
    # flops(G): 310.928
    # params: 243.252K

    # experiments/100_train_DIPNet_x2_m4c16_prelu_RGB/models/net_g_137600.pth (1, 3, 1080, 1920) 2x
    # flops(G): 983.728
    # params: 228.996K

    # experiments/100_train_DIPNet_x2_m4c16_prelu_RGB_mini/models/net_g_108800.pth (1, 3, 1080, 1920) 2x
    # flops(G): 722.327
    # params: 173.046K

    # experiments/100_train_DIPNet_x2_m4c16_prelu_RGB_noSE/models/net_g_227200.pth (1, 3, 1080, 1920) 2x
    # flops(G): 937.617
    # params: 203.476K

    # experiments/100_train_DIPNet_x2_m4c16_prelu_RGB_noSE_mini/models/net_g_368000.pth (1, 3, 1080, 1920) 2x
    # flops(G): 431.604
    # params: 93.664K

    # experiments/100_train_DIPNet_x2_m4c16_prelu_RGB_noSE_mini_small/models/net_g_1596800.pth"  (1, 3, 1080, 1920) 2x
    # flops(G): 112.878
    # params: 24.496K


    checkpoint = model_path

    input_shape = (1, 3, 1080, 1920) # nchw
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, input_data)#.eval()
        scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))
        # scripted_model = torch.jit.load(checkpoint.replace(".pth", ".torchscript.pt"))

    # onnx==10.0.0
    # opset_version=10 can not export onnx in pixel_shuffle
    # import onnx
    with torch.no_grad():
        torch.onnx.export(model, input_data, checkpoint.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
        # dynamic_axes= {
        #             "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
        #             "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
        dynamic_axes= {
                    "input": {0: 'batch_size'},
                    "output": {0: 'batch_size'}}
        )