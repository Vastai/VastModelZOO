# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import compressai
from compressai.models.priors import CompressionModel, GaussianConditional
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.zoo import load_state_dict
import torch.nn as nn
import torch
import math
from torch import Tensor
import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../../source_code")
sys.path.append(common_path)

# from common.vastai_elic import VastaiGs

from elic_dynamic_base import VastaiDynamicHsChunk, VastaiDynamicGs

from ELICUtilis.layers import (
    AttentionBlock,
    conv3x3,
    CheckboardMaskedConv2d,
)


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class Quantizer:
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
        self.conv1 = conv1x1(in_ch, in_ch // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch // 2, in_ch // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch // 2, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out


class DynamicElicDecompress(CompressionModel):
    def __init__(
        self,
        hs_model_info,
        gs0_model_info,
        gs_model_info,
        torch_model,
        tensorize_elf_path,
        batch_size=1,
        device_id=0,
        hs_hw_config="",
        gs0_hw_config="",
        gs_hw_config="",
        max_input_size=[[1, 192, 32, 32]],
        N=192,
        M=320,
        num_slices=5,
    ) -> None:
        super().__init__(entropy_bottleneck_channels=192)

        entropy_coder = compressai.available_entropy_coders()[0]
        compressai.set_entropy_coder(entropy_coder)
        self.groups = [0, 16, 16, 32, 64, 192]  # support depth

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
        # self.vastai_g_s = VastaiGs(gs_model_prefix, batch_size, device_id, gs_hw_config)
        # max_input_size -> xxx
        self.vastai_dynamic_gs = VastaiDynamicGs(
            gs0_model_info,
            [[1, 320, 128, 128]],
            gs_model_info,
            [[1, 192, 512, 512]],
            tensorize_elf_path,
            batch_size,
            device_id,
            gs0_hw_config,
            gs_hw_config,
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
            deconv(N, N * 3 // 2),
            nn.ReLU(inplace=True),
            conv3x3(N * 3 // 2, 2 * M),
        )
        # self.vastai_h_s = VastaiHsChunk(hs_model_prefix, batch_size, device_id, hs_hw_config)
        self.vastai_dynamic_hs = VastaiDynamicHsChunk(
            hs_model_info, max_input_size, batch_size, device_id
        )
        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(
                    self.groups[min(1, i) if i > 0 else 0]
                    + self.groups[i if i > 1 else 0],
                    224,
                    stride=1,
                    kernel_size=5,
                ),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=5),
                nn.ReLU(inplace=True),
                conv(128, self.groups[i + 1] * 2, stride=1, kernel_size=5),
            )
            for i in range(1, num_slices)
        )

        self.context_prediction = nn.ModuleList(
            CheckboardMaskedConv2d(
                self.groups[i + 1],
                2 * self.groups[i + 1],
                kernel_size=5,
                padding=2,
                stride=1,
            )
            for i in range(num_slices)
        )

        self.ParamAggregation = nn.ModuleList(
            nn.Sequential(
                conv1x1(
                    640
                    + self.groups[i + 1 if i > 0 else 0] * 2
                    + self.groups[i + 1] * 2,
                    640,
                ),
                nn.ReLU(inplace=True),
                conv1x1(640, 512),
                nn.ReLU(inplace=True),
                conv1x1(512, self.groups[i + 1] * 2),
            )
            for i in range(num_slices)
        )

        self.gaussian_conditional = GaussianConditional(None)

        state_dict = load_state_dict(torch.load(torch_model))
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

        self.num_slices = num_slices
        self.M = M
        self.N = N

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        B, _, _, _ = z_hat.size()

        latent_means, latent_scales = self.vastai_dynamic_hs.process(z_hat.numpy())
        latent_means = torch.from_numpy(latent_means)
        latent_scales = torch.from_numpy(latent_scales)

        y_strings = strings[0]

        ctx_params_anchor = torch.zeros(
            (B, self.M * 2, z_hat.shape[2] * 4, z_hat.shape[3] * 4)
        )
        ctx_params_anchor_split = torch.split(
            ctx_params_anchor, [2 * i for i in self.groups[1:]], 1
        )

        y_hat_slices = []
        for slice_index in range(len(self.groups) - 1):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = (
                    support_slices_ch.chunk(2, 1)
                )

            else:
                support_slices = torch.concat(
                    [y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1
                )
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = (
                    support_slices_ch.chunk(2, 1)
                )
            ##support mean and scale
            support = (
                torch.concat([latent_means, latent_scales], dim=1)
                if slice_index == 0
                else torch.concat(
                    [
                        support_slices_ch_mean,
                        support_slices_ch_scale,
                        latent_means,
                        latent_scales,
                    ],
                    dim=1,
                )
            )
            ### checkboard process 1
            (
                means_anchor,
                scales_anchor,
            ) = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)
            ).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = means_anchor.size()

            means_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            )
            scales_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            )
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor)

            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional.build_indexes(
                scales_anchor_encode
            )
            anchor_strings = y_strings[slice_index][0]
            anchor_quantized = self.gaussian_conditional.decompress(
                anchor_strings, indexes_anchor, means=means_anchor_encode
            )

            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)
            ).chunk(2, 1)

            means_non_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            )
            scales_non_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            )

            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[
                :, :, 0::2, 1::2
            ]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[
                :, :, 1::2, 0::2
            ]

            indexes_non_anchor = self.gaussian_conditional.build_indexes(
                scales_non_anchor_encode
            )
            non_anchor_strings = y_strings[slice_index][1]
            non_anchor_quantized = self.gaussian_conditional.decompress(
                non_anchor_strings, indexes_non_anchor, means=means_non_anchor_encode
            )

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[
                :, :, 0::2, :
            ]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[
                :, :, 1::2, :
            ]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            y_hat_slices.append(y_slice_hat)
        y_hat = torch.cat(y_hat_slices, dim=1)

        import time

        y_dec_start = time.time()
        # x_hat = self.g_s(y_hat).clamp_(0, 1)
        # output = self.vastai_g_s.process(y_hat.detach().numpy())
        output = self.vastai_dynamic_gs.process(y_hat.detach().numpy())
        x_hat = torch.from_numpy(output[0])
        y_dec = time.time() - y_dec_start

        return {"x_hat": x_hat, "time": {"y_dec": y_dec}}
