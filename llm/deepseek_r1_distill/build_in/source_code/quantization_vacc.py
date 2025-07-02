
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

from torch.nn import Linear
from torch.nn.parameter import Parameter

import bz2
import torch
import base64
import ctypes
from transformers.utils import logging

from typing import List
from functools import partial

logger = logging.get_logger(__name__)

class W8A16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, quant_w: torch.Tensor, scale_w: torch.Tensor, weight_bit_width):
        ctx.inp_shape = inp.size()
        ctx.weight_bit_width = weight_bit_width
        out_features = quant_w.size(0)
        inp = inp.contiguous().view(-1, inp.size(-1))
        weight = extract_weight_to_half(quant_w, scale_w, weight_bit_width)
        ctx.weight_shape = weight.size()
        output = inp.mm(weight.t())
        ctx.save_for_backward(inp, quant_w, scale_w)
        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))
    @staticmethod
    def symbolic(
        g: torch._C.Graph,
        input: torch._C.Value,
        quant_w: torch._C.Value,
        scale_w: torch._C.Value,
        weight_bit_width:torch._C.Value,
    ):
        from torch.onnx.symbolic_helper import _get_tensor_sizes, _get_tensor_dim_size
        # print('_get_tensor_sizes(input)===', _get_tensor_sizes(input))
        # print('_get_tensor_sizes(quant_w)===', _get_tensor_sizes(quant_w))
        opr_type = input.type().with_sizes(_get_tensor_sizes(input)[:-1] + [_get_tensor_sizes(quant_w)[0],])
        ret = g.op("Vastai::QuantizedLinearPerChannel", input, quant_w, scale_w).setType(opr_type)
        return ret
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, quant_w, scale_w = ctx.saved_tensors
        weight = extract_weight_to_half(quant_w, scale_w, ctx.weight_bit_width)
        grad_output = grad_output.contiguous().view(-1, weight.size(0))
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(inp)
        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None, None


def extract_weight_to_half(weight: torch.Tensor, scale_list: torch.Tensor, source_bit_width: int):
    assert weight.dtype in [torch.int8]
    if source_bit_width == 8:
        return weight.to(scale_list.dtype) * scale_list[:, None]
    else:
        assert False, "Unsupported bit-width"


class QuantizedLinear(torch.nn.Module):
    def __init__(self, weight_bit_width: int, weight, bias=None, device="cpu", dtype=None, empty_init=False, *args,
                 **kwargs):
        super().__init__()
        self.weight_bit_width = weight_bit_width

        shape = weight.shape

        if weight is None or empty_init:
            self.weight = torch.empty(shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=device)
            self.weight_scale = torch.empty(shape[0], dtype=dtype, device=device)
        else:
            self.weight_scale = weight.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)
            self.weight = torch.round(weight / self.weight_scale[:, None]).to(torch.int8)
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)

        self.weight = Parameter(self.weight.to(device), requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale.to(device), requires_grad=False)
        self.bias = Parameter(bias.to(device), requires_grad=False) if bias is not None else None

    def forward(self, input):
        output = W8A16Linear.apply(input, self.weight, self.weight_scale, self.weight_bit_width)
        if self.bias is not None:
            output = output + self.bias
        return output


def quantize(model, weight_bit_width, empty_init=False, device=None):
    """Replace fp16 linear with quantized linear"""
    for layer in model.model.layers:
        layer.self_attn.q_proj = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.self_attn.q_proj.weight,
            bias=layer.self_attn.q_proj.bias,
            dtype=layer.self_attn.q_proj.weight.dtype,
            device=layer.self_attn.q_proj.weight.device if device is None else device,
            empty_init=empty_init
        )
        layer.self_attn.k_proj = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.self_attn.k_proj.weight,
            bias=layer.self_attn.k_proj.bias,
            dtype=layer.self_attn.k_proj.weight.dtype,
            device=layer.self_attn.k_proj.weight.device if device is None else device,
            empty_init=empty_init
        )
        layer.self_attn.v_proj = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.self_attn.v_proj.weight,
            bias=layer.self_attn.v_proj.bias,
            dtype=layer.self_attn.v_proj.weight.dtype,
            device=layer.self_attn.v_proj.weight.device if device is None else device,
            empty_init=empty_init
        )
        layer.self_attn.o_proj = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.self_attn.o_proj.weight,
            bias=layer.self_attn.o_proj.bias,
            dtype=layer.self_attn.o_proj.weight.dtype,
            device=layer.self_attn.o_proj.weight.device if device is None else device,
            empty_init=empty_init
        )
        layer.mlp.gate_proj= QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.mlp.gate_proj.weight,
            bias=layer.mlp.gate_proj.bias,
            dtype=layer.mlp.gate_proj.weight.dtype,
            device=layer.mlp.gate_proj.weight.device if device is None else device,
            empty_init=empty_init
        )
        layer.mlp.down_proj= QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.mlp.down_proj.weight,
            bias=layer.mlp.down_proj.bias,
            dtype=layer.mlp.down_proj.weight.dtype,
            device=layer.mlp.down_proj.weight.device if device is None else device,
            empty_init=empty_init
        )
        layer.mlp.up_proj= QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.mlp.up_proj.weight,
            bias=layer.mlp.up_proj.bias,
            dtype=layer.mlp.up_proj.weight.dtype,
            device=layer.mlp.up_proj.weight.device if device is None else device,
            empty_init=empty_init
        )

    return model
