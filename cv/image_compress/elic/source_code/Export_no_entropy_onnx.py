# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
A simple test algorithm to rewrite the network
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
from timm.models.layers import trunc_normal_
from ELICUtilis.layers import (
    AttentionBlock,
    conv3x3,
    CheckboardMaskedConv2d,
)

from compressai.models.priors import CompressionModel, GaussianConditional
from compressai.ops import ste_round
from compressai.models.utils import conv, deconv, update_registered_buffers

from thop import profile
from ptflops import get_model_complexity_info

from compressai.zoo import load_state_dict

import numpy as np

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

class ResidualBottleneckBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch//2, in_ch//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch//2, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out


class Quantizer():
    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)

class TestModel(CompressionModel):

    def __init__(self, N=192, M=320, num_slices=5, **kwargs):
        super().__init__(entropy_bottleneck_channels=192)
        self.N = int(N)
        self.M = int(M)
        self.num_slices = num_slices

        """
             N: channel number of main network
             M: channnel number of latent space
             
        """
        self.groups = [0, 16, 16, 32, 64, 192] #support depth
        self.g_a = nn.Sequential(
            conv(3, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv(N, M),
            AttentionBlock(M),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, N),
            AttentionBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N*3//2),
            nn.ReLU(inplace=True),
            conv3x3(N*3//2, 2*M),
        )

        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.groups[min(1, i) if i > 0 else 0] + self.groups[i if i > 1 else 0], 224, stride=1,
                     kernel_size=5),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=5),
                nn.ReLU(inplace=True),
                conv(128, self.groups[i + 1]*2, stride=1, kernel_size=5),
            ) for i in range(1,  num_slices)
        ) ## from https://github.com/tensorflow/compression/blob/master/models/ms2020.py

        self.context_prediction = nn.ModuleList(
            CheckboardMaskedConv2d(
            self.groups[i+1], 2*self.groups[i+1], kernel_size=5, padding=2, stride=1
            ) for i in range(num_slices)
        )## from https://github.com/JiangWeibeta/Checkerboard-Context-Model-for-Efficient-Learned-Image-Compression/blob/main/version2/layers/CheckerboardContext.py

        self.ParamAggregation = nn.ModuleList(
            nn.Sequential(
                conv1x1(640 + self.groups[i+1 if i > 0 else 0] * 2 + self.groups[
                        i + 1] * 2, 640),
                nn.ReLU(inplace=True),
                conv1x1(640, 512),
                nn.ReLU(inplace=True),
                conv1x1(512, self.groups[i + 1]*2),
            ) for i in range(num_slices)
        ) ##from checkboard "Checkerboard Context Model for Efficient Learned Image Compression"" gep网络参数

        self.quantizer = Quantizer()

        self.gaussian_conditional = GaussianConditional(None)


    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    # forward origin
    def forward_ori(self, x, noisequant=False):
        y = self.g_a(x)
        B, C, H, W = y.size() ## The shape of y to generate the mask

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.groups[1:], 1)

        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device),
                                              [2 * i for i in self.groups[1:]], 1)
        y_hat_slices = []
        y_hat_slices_for_gs = []
        y_likelihood = []
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index-1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index-1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index-1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            y_anchor = anchor_split[slice_index]
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            scales_hat_split = torch.zeros_like(y_anchor).to(x.device)
            means_hat_split = torch.zeros_like(y_anchor).to(x.device)

            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]
            if noisequant:
                y_anchor_quantilized = self.quantizer.quantize(y_anchor, "noise")
                y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor, "ste")
            else:
                y_anchor_quantilized = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor
                y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor


            y_anchor_quantilized[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized[:, :, 1::2, 0::2] = 0
            y_anchor_quantilized_for_gs[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized_for_gs[:, :, 1::2, 0::2] = 0

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_quantilized)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
            # entropy estimation
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat_split, means=means_hat_split)

            y_non_anchor = non_anchor_split[slice_index]
            if noisequant:
                y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor, "noise")
                y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor, "ste")
            else:
                y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                          "ste") + means_non_anchor
                y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                          "ste") + means_non_anchor

            y_non_anchor_quantilized[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized[:, :, 1::2, 1::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_quantilized + y_non_anchor_quantilized
            y_hat_slice_for_gs = y_anchor_quantilized_for_gs + y_non_anchor_quantilized_for_gs
            y_hat_slices.append(y_hat_slice)
            ### ste for synthesis model
            y_hat_slices_for_gs.append(y_hat_slice_for_gs)
            y_likelihood.append(y_slice_likelihood)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        """
        use STE(y) as the input of synthesizer
        """
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # remove likelihoods
    def forward_remove_likelihoods(self, x, noisequant=False):
        y = self.g_a(x)
        B, C, H, W = y.size() ## The shape of y to generate the mask

        z = self.h_a(y)
        z_hat, _ = self.entropy_bottleneck(z)
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.groups[1:], 1)

        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device),
                                              [2 * i for i in self.groups[1:]], 1)
        y_hat_slices = []
        y_hat_slices_for_gs = []
        # y_likelihood = []
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index-1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index-1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index-1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            y_anchor = anchor_split[slice_index]
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            scales_hat_split = torch.zeros_like(y_anchor).to(x.device)
            # means_hat_split = torch.zeros_like(y_anchor).to(x.device)

            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            # means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            # means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]
            if noisequant:
                y_anchor_quantilized = self.quantizer.quantize(y_anchor, "noise")
                y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor, "ste")
            else:
                y_anchor_quantilized = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor
                y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor


            y_anchor_quantilized[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized[:, :, 1::2, 0::2] = 0
            y_anchor_quantilized_for_gs[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized_for_gs[:, :, 1::2, 0::2] = 0

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_quantilized)
            means_non_anchor, _ = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            # scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            # scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            # means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            # means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
            # # entropy estimation
            # _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat_split, means=means_hat_split)

            y_non_anchor = non_anchor_split[slice_index]
            if noisequant:
                y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor, "noise")
                y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor, "ste")
            else:
                y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                          "ste") + means_non_anchor
                y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                          "ste") + means_non_anchor

            y_non_anchor_quantilized[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized[:, :, 1::2, 1::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_quantilized + y_non_anchor_quantilized
            y_hat_slice_for_gs = y_anchor_quantilized_for_gs + y_non_anchor_quantilized_for_gs
            y_hat_slices.append(y_hat_slice)
            ### ste for synthesis model
            y_hat_slices_for_gs.append(y_hat_slice_for_gs)
            # y_likelihood.append(y_slice_likelihood)

        # y_likelihoods = torch.cat(y_likelihood, dim=1)
        """
        use STE(y) as the input of synthesizer
        """
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            # "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # compress forward
    def forward_compress(self, x):
        y = self.g_a(x)
        B, C, H, W = y.size()  ## The shape of y to generate the mask

        z = self.h_a(y)
        
        z_strings = self.entropy_bottleneck.compress(z)
        
        # return torch.frombuffer(z_strings[0], dtype=torch.uint8)
        import io
        # 使用io.BytesIO将字节流包装成文件对象
        model_bytes_io = io.BytesIO(z_strings[0])
        
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        y_slices = torch.split(y, self.groups[1:], 1)

        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device), [2 * i for i in self.groups[1:]], 1)

        # added by test
        temp_anchor_strings = [] 
        temp_non_anchor_strings = []
        
        y_strings = []
        y_hat_slices = []
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            y_anchor = y_slices[slice_index].clone()
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = y_anchor.size()

            y_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(x.device)

            y_anchor_encode[:, :, 0::2, :] = y_anchor[:, :, 0::2, 0::2]
            y_anchor_encode[:, :, 1::2, :] = y_anchor[:, :, 1::2, 1::2]
            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = self.gaussian_conditional.compress(y_anchor_encode, indexes_anchor, means=means_anchor_encode)
            anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor, means=means_anchor_encode)
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]


            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            y_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)

            non_anchor = y_slices[slice_index].clone()
            y_non_anchor_encode[:, :, 0::2, :] = non_anchor[:, :, 0::2, 1::2]
            y_non_anchor_encode[:, :, 1::2, :] = non_anchor[:, :, 1::2, 0::2]
            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]

            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor_encode, indexes_non_anchor,
                                                                    means=means_non_anchor_encode)

            non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
                                                                        means=means_non_anchor_encode)

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            y_hat_slices.append(y_slice_hat)

            # # added by debug
            # for anchor_temp in anchor_strings:
            #     # temp_anchor_strings.append(io.BytesIO(anchor_temp))
            #     temp_anchor_strings.append(str(anchor_temp))
                
            
            # for non_anchor_temp in non_anchor_strings:
            #     # temp_non_anchor_strings.append(io.BytesIO(non_anchor_temp))
            #     temp_non_anchor_strings.append(str(non_anchor_temp))
            
            # y_strings.append([temp_anchor_strings, temp_non_anchor_strings])
            y_strings.append([anchor_strings, non_anchor_strings])
        
        # # add by test
        # temp_z_strings = []
        # for z_temp in z_strings:
        #     # temp_z_strings.append(io.BytesIO(z_temp))
        #     temp_z_strings.append(str(z_temp))
        # return {"shape": z.size()[-2:]}
        # return {"strings": []}
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    # inference forward
    def forward_inference(self, x):
        # # save result
        # import numpy as np
        # out_net["x_hat"].detach().numpy().astype(np.float16).tofile("./result/torch_out_x_hat_1080x1920.bin")
        # out_net["likelihoods"]["y"].detach().numpy().astype(np.float16).tofile("./result/torch_out_y_likelihoods_1080x1920.bin")
        # out_net["likelihoods"]["z"].detach().numpy().astype(np.float16).tofile("./result/torch_out_z_likelihoods_1080x1920.bin")
        # return

        y = self.g_a(x)
        B, C, H, W = y.size()  ## The shape of y to generate the mask
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        # generate numpy constant
        
        # anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        # anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        # non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        # non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.groups[1:], 1)

        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device),
                                              [2 * i for i in self.groups[1:]], 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            y_anchor = anchor_split[slice_index]
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            scales_hat_split = torch.zeros_like(y_anchor).to(x.device)
            means_hat_split = torch.zeros_like(y_anchor).to(x.device)

            # scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            # scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            # means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            # means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]

            y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor

            # y_anchor_quantilized_for_gs[:, :, 0::2, 1::2] = 0
            # y_anchor_quantilized_for_gs[:, :, 1::2, 0::2] = 0

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_quantilized_for_gs)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            # scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            # scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            # means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            # means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
            
            # entropy estimation
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat_split, means=means_hat_split)

            y_non_anchor = non_anchor_split[slice_index]

            y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                      "ste") + means_non_anchor
            # y_non_anchor_quantilized_for_gs[:, :, 0::2, 0::2] = 0
            # y_non_anchor_quantilized_for_gs[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_quantilized_for_gs + y_non_anchor_quantilized_for_gs
            y_hat_slices.append(y_hat_slice)
            ### ste for synthesis model
            y_likelihood.append(y_slice_likelihood)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        """
        use STE(y) as the input of synthesizer
        """
        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            # "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        }

    def generate_numpy_constant_replace_scatternd(self, src, dst, device, h_index, w_index, step):
        torch_help_array = torch.zeros_like(src).to(device)
        help_array = torch_help_array.detach().numpy() # from torch.tensor to numpy
        help_array[:, :, int(h_index)::int(step), int(w_index)::int(step)] = 1
        torch_help_array = torch.from_numpy(help_array)  # from numpy to torch.tensor
        dst = dst + src * torch_help_array
        return dst
      
        
    # rewriter inference forward
    def forward(self, x):
        print("rewriter inference forward...")
        # # save result
        # import numpy as np
        # out_net["x_hat"].detach().numpy().astype(np.float16).tofile("./result/torch_out_x_hat_1080x1920.bin")
        # out_net["likelihoods"]["y"].detach().numpy().astype(np.float16).tofile("./result/torch_out_y_likelihoods_1080x1920.bin")
        # out_net["likelihoods"]["z"].detach().numpy().astype(np.float16).tofile("./result/torch_out_z_likelihoods_1080x1920.bin")
        # return

        y = self.g_a(x)
        B, C, H, W = y.size()  ## The shape of y to generate the mask
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        # # add <begin> generate numpy constant
        anchor = self.generate_numpy_constant_replace_scatternd(y, anchor, x.device, 0, 0, 2)
        anchor = self.generate_numpy_constant_replace_scatternd(y, anchor, x.device, 1, 1, 2)
        non_anchor = self.generate_numpy_constant_replace_scatternd(y, non_anchor, x.device, 0, 1, 2)
        non_anchor = self.generate_numpy_constant_replace_scatternd(y, non_anchor, x.device, 1, 0, 2)
        # # add <end> 
        
        # anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        # anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        # non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        # non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.groups[1:], 1)

        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device),
                                              [2 * i for i in self.groups[1:]], 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            y_anchor = anchor_split[slice_index]
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            scales_hat_split = torch.zeros_like(y_anchor).to(x.device)
            means_hat_split = torch.zeros_like(y_anchor).to(x.device)

            # # add <begin> generate numpy constant
            scales_hat_split = self.generate_numpy_constant_replace_scatternd(scales_anchor, scales_hat_split, x.device, 0, 0, 2)
            scales_hat_split = self.generate_numpy_constant_replace_scatternd(scales_anchor, scales_hat_split, x.device, 1, 1, 2)
            means_hat_split = self.generate_numpy_constant_replace_scatternd(means_anchor, means_hat_split, x.device, 0, 0, 2)
            means_hat_split = self.generate_numpy_constant_replace_scatternd(means_anchor, means_hat_split, x.device, 1, 1, 2)
            # # add <end> 
            
            # scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            # scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            # means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            # means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]

            y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor

            # # add <begin> generate numpy constant
            torch_ones_tensor = torch.ones_like(y_anchor_quantilized_for_gs).to(x.device)
            help_array = torch_ones_tensor.detach().numpy()
            help_array[:, :, 0::2, 1::2] = 0
            help_array[:, :, 1::2, 0::2] = 0
            y_anchor_quantilized_for_gs = y_anchor_quantilized_for_gs * torch.from_numpy(help_array)
            # # add <end> 
            
            # y_anchor_quantilized_for_gs[:, :, 0::2, 1::2] = 0
            # y_anchor_quantilized_for_gs[:, :, 1::2, 0::2] = 0

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_quantilized_for_gs)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            # # add <begin> generate numpy constant
            # scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat_one_tensor = torch.ones_like(scales_hat_split).to(x.device).detach().numpy()
            scales_hat_one_tensor[:, :, 0::2, 1::2] = 0
            scales_hat_split = scales_hat_split * torch.from_numpy(scales_hat_one_tensor)
            
            scales_non_anchor_zero = torch.zeros_like(scales_non_anchor).to(x.device).detach().numpy()
            scales_non_anchor_zero[:, :, 0::2, 1::2] = 1
            scales_hat_split = scales_hat_split + scales_non_anchor * torch.from_numpy(scales_non_anchor_zero)
            
            # scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            scales_hat_one_tensor1 = torch.ones_like(scales_hat_split).to(x.device).detach().numpy()
            scales_hat_one_tensor1[:, :, 1::2, 0::2] = 0
            scales_hat_split = scales_hat_split * torch.from_numpy(scales_hat_one_tensor1)
            
            scales_non_anchor_zero1 = torch.zeros_like(scales_non_anchor).to(x.device).detach().numpy()
            scales_non_anchor_zero1[:, :, 1::2, 0::2] = 1
            scales_hat_split = scales_hat_split + scales_non_anchor * torch.from_numpy(scales_non_anchor_zero1)
            
            # mean
            # means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            
            means_hat_one_tensor = torch.ones_like(means_hat_split).to(x.device).detach().numpy()
            means_hat_one_tensor[:, :, 0::2, 1::2] = 0
            means_hat_split = means_hat_split * torch.from_numpy(means_hat_one_tensor)
            
            means_non_anchor_zero = torch.zeros_like(means_non_anchor).to(x.device).detach().numpy()
            means_non_anchor_zero[:, :, 0::2, 1::2] = 1
            means_hat_split = means_hat_split + means_non_anchor * torch.from_numpy(means_non_anchor_zero)
            
            # means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
            means_hat_one_tensor1 = torch.ones_like(means_hat_split).to(x.device).detach().numpy()
            means_hat_one_tensor1[:, :, 1::2, 0::2] = 0
            means_hat_split = means_hat_split * torch.from_numpy(means_hat_one_tensor1)
            
            means_non_anchor_zero1 = torch.zeros_like(means_non_anchor).to(x.device).detach().numpy()
            means_non_anchor_zero1[:, :, 1::2, 0::2] = 1
            means_hat_split = means_hat_split + means_non_anchor * torch.from_numpy(means_non_anchor_zero1)
            
            # # add <end>
            
            # scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            # scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            # means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            # means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
            
            # entropy estimation
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat_split, means=means_hat_split)

            y_non_anchor = non_anchor_split[slice_index]

            y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                      "ste") + means_non_anchor
            
            # # add <begin> generate numpy constant
            torch_ones_non_anchor = torch.ones_like(y_non_anchor_quantilized_for_gs).to(x.device)
            help_array_1 = torch_ones_non_anchor.detach().numpy()
            help_array_1[:, :, 0::2, 0::2] = 0
            help_array_1[:, :, 1::2, 1::2] = 0
            y_non_anchor_quantilized_for_gs = y_non_anchor_quantilized_for_gs * torch.from_numpy(help_array_1)
            # # add <end> 
            
            # y_non_anchor_quantilized_for_gs[:, :, 0::2, 0::2] = 0
            # y_non_anchor_quantilized_for_gs[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_quantilized_for_gs + y_non_anchor_quantilized_for_gs
            y_hat_slices.append(y_hat_slice)
            ### ste for synthesis model
            y_likelihood.append(y_slice_likelihood)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        """
        use STE(y) as the input of synthesizer
        """
        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        }
        
    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        import time
        y_enc_start = time.time()
        y = self.g_a(x)
        y_enc = time.time() - y_enc_start
        B, C, H, W = y.size()  ## The shape of y to generate the mask

        z_enc_start = time.time()
        z = self.h_a(y) # torch.Size([1, 192, 8, 8])
        z_enc = time.time() - z_enc_start
        
        compress_start = time.time()
        ## odsp start
        z_strings = self.entropy_bottleneck.compress(z) # host
        # [b'\xb9\xc6\xf6\xd4\r<P\x07\xc6\xb6\x0b\xb3!\x94\x11\xe9\xb2\xce\x94Uy&ve\xb6#OA\x1193\x94\xfd\x1d\xd8\x9e\xa3\x93\x82\xb6}\x16o\x85^\'\xc61\xd7\n\xbcv%\xb2\xf6\xc9\xd6\x1f\xc2\xd0\x94B\xe8\xc4}\x15Wh\xd91\xe2\x89MhK\xb4\xbd\x94\xd6\nS\xa6g~\xa0\x8d\x1dr\xe9}\x08\x11\x80EL\x86\x9e"\x80\x13\xb3[\x034\xae\x19\xc6\xd5\xcbRgJ\x95f\xe7\x91\xf9G=!\xce`M"\xe4\x13\xa5k\x84\x9d\x98\xde\xb7\xf8\x92M\x99\xfb\xbcS\xeb\x16\xe4\xecU\x01\xff\xec \x01M\x0f\x10,\x98\xda\x13\r\xb6}\xd8\x12\xd3\xd1"|\xff!\x15\x940\xd1\xd1\xac;\xac9)\xa6\r\x9a\xe8\xe0Q\xd0\xfb\x0e\xe6J\xc8m\xaf\x1e\xa7\x0e$@\x05\xcb\xd2\xe1\x08\x12\x84\xf2\x99\xe8\xbe\x8e\xd7\xcc\xa0?D\xd1\xac\xd8\xb7\x00y\x9a\x98\xe2`\xe2\xd2Z\x10\x88 Y2\xa4\x96#\xed\x99\xf2\xd3A\x85(qhd\xe9+Y\x1b\x9f\xea&\xe9\xfd\x00\xf6hWD@h\xd8_\x85E\xac!\xcdB0m\xe5\xae\xc7\xfe\xe5\xf4\x00\x8a\x1b\xef\xafl\xd5KO\x8b\xcc\x8b,&\x95>\x19\xc1y\x18\xba\x90\x92y\xf0D\x8a*\xf7),\xf7\xaf\xce@\xc8#n\xb3\xcf\xb9J4\x16\x13\xdb\x1b\xeb\xd7\xf1Oy\xfb\\d\xef)\xf7\x9d\x89\n\xae\xb4\xc26\x80\x01']
        # string_object = z_strings.decode('utf-8')
        # print("ztrings string_object: ", z_strings)
        
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:]) # host # torch.Size([1, 192, 8, 8])
        com_decom_time = time.time() - compress_start
        print("com_decomn time: ", com_decom_time)
        ## odsp end
        
        z_dec_start = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec_start

        y_slices = torch.split(y, self.groups[1:], 1)

        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device), [2 * i for i in self.groups[1:]], 1)

        y_strings = []
        y_hat_slices = []
        params_start = time.time()
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            y_anchor = y_slices[slice_index].clone()
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = y_anchor.size()

            y_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(x.device)

            y_anchor_encode[:, :, 0::2, :] = y_anchor[:, :, 0::2, 0::2]
            y_anchor_encode[:, :, 1::2, :] = y_anchor[:, :, 1::2, 1::2]
            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            ## odsp start
            for_compress_time = time.time()
            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = self.gaussian_conditional.compress(y_anchor_encode, indexes_anchor, means=means_anchor_encode) # host
            anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor, means=means_anchor_encode) # host
            
            for_compress1_time = time.time() - for_compress_time
            print("for_compress1_time: ", for_compress1_time)
            ## odsp end
            
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]


            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            y_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)

            non_anchor = y_slices[slice_index].clone()
            y_non_anchor_encode[:, :, 0::2, :] = non_anchor[:, :, 0::2, 1::2]
            y_non_anchor_encode[:, :, 1::2, :] = non_anchor[:, :, 1::2, 0::2]
            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]

            ## odsp start
            for_compress_time1 = time.time()
            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor_encode, indexes_non_anchor,
                                                                    means=means_non_anchor_encode) # host

            non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
                                                                        means=means_non_anchor_encode) # host
            for_compress2_time = time.time() - for_compress_time1
            print("for_compress2_time: ", for_compress2_time)
            ## odsp end
            
            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            y_hat_slices.append(y_slice_hat)

            y_strings.append([anchor_strings, non_anchor_strings])

        params_time = time.time() - params_start
        print("params_time: ", params_time)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],
                "time": {'y_enc': y_enc, "z_enc": z_enc, "z_dec": z_dec, "params": params_time}}

    def compress_rewriter(self, x):
        import time
        y_enc_start = time.time()
        y = self.g_a(x)
        y_enc = time.time() - y_enc_start
        B, C, H, W = y.size()  ## The shape of y to generate the mask

        z_enc_start = time.time()
        z = self.h_a(y) # torch.Size([1, 192, 8, 8])
        z_enc = time.time() - z_enc_start
        z_strings = self.entropy_bottleneck.compress(z) # host
        # [b'\xb9\xc6\xf6\xd4\r<P\x07\xc6\xb6\x0b\xb3!\x94\x11\xe9\xb2\xce\x94Uy&ve\xb6#OA\x1193\x94\xfd\x1d\xd8\x9e\xa3\x93\x82\xb6}\x16o\x85^\'\xc61\xd7\n\xbcv%\xb2\xf6\xc9\xd6\x1f\xc2\xd0\x94B\xe8\xc4}\x15Wh\xd91\xe2\x89MhK\xb4\xbd\x94\xd6\nS\xa6g~\xa0\x8d\x1dr\xe9}\x08\x11\x80EL\x86\x9e"\x80\x13\xb3[\x034\xae\x19\xc6\xd5\xcbRgJ\x95f\xe7\x91\xf9G=!\xce`M"\xe4\x13\xa5k\x84\x9d\x98\xde\xb7\xf8\x92M\x99\xfb\xbcS\xeb\x16\xe4\xecU\x01\xff\xec \x01M\x0f\x10,\x98\xda\x13\r\xb6}\xd8\x12\xd3\xd1"|\xff!\x15\x940\xd1\xd1\xac;\xac9)\xa6\r\x9a\xe8\xe0Q\xd0\xfb\x0e\xe6J\xc8m\xaf\x1e\xa7\x0e$@\x05\xcb\xd2\xe1\x08\x12\x84\xf2\x99\xe8\xbe\x8e\xd7\xcc\xa0?D\xd1\xac\xd8\xb7\x00y\x9a\x98\xe2`\xe2\xd2Z\x10\x88 Y2\xa4\x96#\xed\x99\xf2\xd3A\x85(qhd\xe9+Y\x1b\x9f\xea&\xe9\xfd\x00\xf6hWD@h\xd8_\x85E\xac!\xcdB0m\xe5\xae\xc7\xfe\xe5\xf4\x00\x8a\x1b\xef\xafl\xd5KO\x8b\xcc\x8b,&\x95>\x19\xc1y\x18\xba\x90\x92y\xf0D\x8a*\xf7),\xf7\xaf\xce@\xc8#n\xb3\xcf\xb9J4\x16\x13\xdb\x1b\xeb\xd7\xf1Oy\xfb\\d\xef)\xf7\x9d\x89\n\xae\xb4\xc26\x80\x01']
        # string_object = z_strings.decode('utf-8')
        # print("ztrings string_object: ", z_strings)
        
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:]) # host # torch.Size([1, 192, 8, 8])

        z_dec_start = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec_start

        y_slices = torch.split(y, self.groups[1:], 1)

        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device), [2 * i for i in self.groups[1:]], 1)

        y_strings = []
        y_hat_slices = []
        params_start = time.time()
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            y_anchor = y_slices[slice_index].clone()
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = y_anchor.size()

            y_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(x.device)

            y_anchor_encode[:, :, 0::2, :] = y_anchor[:, :, 0::2, 0::2]
            y_anchor_encode[:, :, 1::2, :] = y_anchor[:, :, 1::2, 1::2]
            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = self.gaussian_conditional.compress(y_anchor_encode, indexes_anchor, means=means_anchor_encode) # host
            anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor, means=means_anchor_encode) # host
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]


            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            y_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)

            non_anchor = y_slices[slice_index].clone()
            y_non_anchor_encode[:, :, 0::2, :] = non_anchor[:, :, 0::2, 1::2]
            y_non_anchor_encode[:, :, 1::2, :] = non_anchor[:, :, 1::2, 0::2]
            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]

            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor_encode, indexes_non_anchor,
                                                                    means=means_non_anchor_encode) # host

            # non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
            #                                                             means=means_non_anchor_encode) # host

            # y_non_anchor_quantized = torch.zeros_like(means_anchor)
            # y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            # y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            # y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            # y_hat_slices.append(y_slice_hat)

            y_strings.append([anchor_strings, non_anchor_strings])

        params_time = time.time() - params_start
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],
                "time": {'y_enc': y_enc, "z_enc": z_enc, "z_dec": z_dec, "params": params_time}}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        # FIXME: we don't respect the default entropy coder and directly call thse
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        B, _, _, _ = z_hat.size()

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_strings = strings[0]

        ctx_params_anchor = torch.zeros((B, self.M*2, z_hat.shape[2] * 4, z_hat.shape[3] * 4)).to(z_hat.device)
        ctx_params_anchor_split = torch.split(ctx_params_anchor, [2 * i for i in self.groups[1:]], 1)


        y_hat_slices = []
        for slice_index in range(len(self.groups) - 1):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = means_anchor.size()

            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(z_hat.device)

            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = y_strings[slice_index][0]
            anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor,
                                                                    means=means_anchor_encode)

            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)

            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]

            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = y_strings[slice_index][1]
            non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
                                                                        means=means_non_anchor_encode)

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            y_hat_slices.append(y_slice_hat)
        y_hat = torch.cat(y_hat_slices, dim=1)

        import time
        y_dec_start = time.time()
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        y_dec = time.time() - y_dec_start

        return {"x_hat": x_hat, "time":{"y_dec": y_dec}}

    def inference(self, x):
        import time
        y_enc_start = time.time()
        y = self.g_a(x)
        y_enc = time.time() - y_enc_start
        B, C, H, W = y.size()  ## The shape of y to generate the mask

        z_enc_start = time.time()
        z = self.h_a(y)
        z_enc = time.time() - z_enc_start
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        z_dec_start = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec_start

        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.groups[1:], 1)

        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device),
                                              [2 * i for i in self.groups[1:]], 1)
        y_hat_slices = []
        y_likelihood = []
        params_start = time.time()
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            y_anchor = anchor_split[slice_index]
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            scales_hat_split = torch.zeros_like(y_anchor).to(x.device)
            means_hat_split = torch.zeros_like(y_anchor).to(x.device)

            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]

            y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor

            y_anchor_quantilized_for_gs[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized_for_gs[:, :, 1::2, 0::2] = 0

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_quantilized_for_gs)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
            # entropy estimation
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat_split, means=means_hat_split)

            y_non_anchor = non_anchor_split[slice_index]

            y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                      "ste") + means_non_anchor
            y_non_anchor_quantilized_for_gs[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_quantilized_for_gs + y_non_anchor_quantilized_for_gs
            y_hat_slices.append(y_hat_slice)
            ### ste for synthesis model
            y_likelihood.append(y_slice_likelihood)

        params_time = time.time() - params_start
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        """
        use STE(y) as the input of synthesizer
        """
        y_hat = torch.cat(y_hat_slices, dim=1)
        y_dec_start = time.time()
        x_hat = self.g_s(y_hat)
        y_dec = time.time() - y_dec_start
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "time": {'y_enc': y_enc, "y_dec": y_dec, "z_enc": z_enc, "z_dec": z_dec, "params":params_time}
        }


# added for exporting onnx model
def export_onnx(input, in_path, out_path):
    # import onnxscript
    # from onnxscript.onnx_opset import opset12 as op
    # from torch.onnx._internal import jit_utils
    
    opset_version = 12
    
    # custom_opset = onnxscript.values.Opset(domain="torch.onnx", version=1)
    
    # @onnxscript.script(custom_opset)
    # def erfc(x):
    #     # y=op.Sub(1, op.Erf(x))
    #     y = 1 - op.erf(x)
    #     return y

    # def custom_erfc(g: jit_utils.GraphContext, x):
    #     return g.onnxscript_op(erfc).setType(x.type())

    # torch.onnx.register_custom_op_symbolic(
    # symbolic_name="aten::erfc",
    # symbolic_fn=custom_erfc,
    # opset_version=opset_version,
    # )
        
    _, _, h, w = input.shape
    # export onnx
    state_dict = load_state_dict(torch.load(in_path))
    model_cls = TestModel()
    model = model_cls.from_state_dict(state_dict).eval()
    print("model: ", model)


    import os
    pth_path = os.path.abspath(in_path) #绝对路径
    dir_name, file_name = os.path.split(pth_path) # split dir and file
    model_name = file_name.split(".")[0]
    # model.eval()
    # mode = '_ori_'
    mode = '_origin_remove_scatterND_'
    # mode = '_compress_'
    model_name = model_name + mode + str(h) + 'x' + str(w)
    # onnx_model_path = model_name + '.onnx'
    onnx_model_path = out_path
    print("onnx_model_path: ", onnx_model_path )
    
    print('export onnx model begin...')
    output_names = ['x_hat', 'y_likelihoods', 'z_likelihoods']
    # output_names = ['x_hat']
    # output_names = ['y_strings', 'z_strings', 'shape']
    # return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
    # output_names = ['shape']
    
    try: 
        torch.onnx.export(model, input, 
                        onnx_model_path, 
                        opset_version=opset_version, 
                        input_names = ['input'],
                        output_names = output_names,
                        do_constant_folding=True, verbose=False) #opset_version=15, input_names=['input'], 
    except UnicodeDecodeError:
        print("UnicodeDecodeError !") 
    
    onnxsim(onnx_model_path, model_name, "./")
    
    print('export onnx model done...')

def onnxsim(model_onnx, model_name, model_dir, shape_dict={}):
    import onnxsim
    import onnx

    import os

    f = os.path.join(os.path.abspath(model_dir), "{}_sim.onnx".format(model_name))  
    print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
    if len(shape_dict) == 0:
        model_onnx, check = onnxsim.simplify(model_onnx, dynamic_input_shape=False)
    else:
        model_onnx, check = onnxsim.simplify(model_onnx, dynamic_input_shape=False, overwrite_input_shapes=shape_dict)
    assert check, 'assert check failed'
    onnx.save(model_onnx, f)
    print("onnxsim done")

import argparse

def parse_int_list(value):
    return [int(x) for x in value.split(',')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export no entropy model")
    parser.add_argument("--pth_model",type=str,default="./ELIC_0450_ft_3980_Plateau.pth.tar")
    parser.add_argument("--model_input_shape", type = parse_int_list, default = [1,3, 512, 512], help = "model input shape")
    parser.add_argument("--onnx_out_path",type=str,default="./elic_no_entropy_512x512.onnx",help="model info")

    args = parser.parse_args()

    # input = torch.randn(1, 3, 1280, 2048)
    shape = args.model_input_shape
    input = torch.randn(shape[0], shape[1], shape[2], shape[3])
    in_path = args.pth_model
    out_path = args.onnx_out_path
    export_onnx(input, in_path, out_path)