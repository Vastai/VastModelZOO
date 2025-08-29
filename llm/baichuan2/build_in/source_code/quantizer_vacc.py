# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import bitsandbytes as bnb
from torch.nn.parameter import Parameter
from torch.nn import Linear
import bz2
from accelerate import init_empty_weights
from bitsandbytes.nn.modules import Params4bit, Int8Params
import torch 

def Params4bitCuda(self, device):
    self.data = self.data.cuda(device)
    self.quant_state[0] = self.quant_state[0].cuda(device)
    self.quant_state[4][0] = self.quant_state[4][0].cuda(device)
    self.quant_state[4][1][0] = self.quant_state[4][1][0].cuda(device)
    self.quant_state[4][1][1] = self.quant_state[4][1][1].cuda(device)

    self.quant_state[6] = self.quant_state[6].cuda(device)
    return self

class Linear4bitOnline(torch.nn.Module):
    def __init__(self, weight, bias, quant_type):
        super().__init__()
        self.weight = Params4bit(
            weight.data, requires_grad=False, compress_statistics=True, quant_type=quant_type
        )
        self.compute_dtype = None
        #self.weight.cuda(weight.device)
        self.bias = bias

    def forward(self, x: torch.Tensor):
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if getattr(self.weight, "quant_state", None) is None:
            print(
                "FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first."
            )
        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        out = bnb.matmul_4bit(
            x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state
        )

        out = out.to(inp_dtype)

        return out
    
class Linear8bitLtOnline(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
        has_fp16_weights=True,
        memory_efficient_backward=False,
        threshold=0.0,
        index=None,
    ):
        super().__init__()
        assert (
            not memory_efficient_backward
        ), "memory_efficient_backward is no longer required and the argument is deprecated in 0.37.0 and will be removed in 0.39.0"
        self.state = bnb.MatmulLtState()
        self.index = index

        # Necessary for stacked layers
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = Int8Params(
            weight.data,
            has_fp16_weights=has_fp16_weights,
            requires_grad=has_fp16_weights,
        )
        self.bias = bias

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x: torch.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)
        
        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.data = self.state.CxB
        return out
    
def quantize_offline(model, bits: int):
    assert (bits == 4), f'bits: {bits} is not supported'
    
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.W_pack = bnb.nn.Linear4bit(
                            layer.self_attn.W_pack.weight.shape[1],
                            layer.self_attn.W_pack.weight.shape[0],
                            False,
                            torch.float16,
                            compress_statistics=True,
                            quant_type="nf4",
                        )
        layer.self_attn.o_proj = bnb.nn.Linear4bit(
                            layer.self_attn.o_proj.weight.shape[1],
                            layer.self_attn.o_proj.weight.shape[0],
                            False,
                            torch.float16,
                            compress_statistics=True,
                            quant_type="nf4",
                        )

        layer.mlp.gate_proj = bnb.nn.Linear4bit(
                            layer.mlp.gate_proj.weight.shape[1],
                            layer.mlp.gate_proj.weight.shape[0],
                            False,
                            torch.float16,
                            compress_statistics=True,
                            quant_type="nf4",
                        )
        layer.mlp.down_proj = bnb.nn.Linear4bit(
                            layer.mlp.down_proj.weight.shape[1],
                            layer.mlp.down_proj.weight.shape[0],
                            False,
                            torch.float16,
                            compress_statistics=True,
                            quant_type="nf4",
                        )
        layer.mlp.up_proj = bnb.nn.Linear4bit(
                            layer.mlp.up_proj.weight.shape[1],
                            layer.mlp.up_proj.weight.shape[0],
                            False,
                            torch.float16,
                            compress_statistics=True,
                            quant_type="nf4",
                        )
    return model

def quantize_online(model, bits: int):
    def quant(weight, bias=None):
        if bits == 8:
            linear = Linear8bitLtOnline(
                weight,
                bias,
                has_fp16_weights=False,
                threshold=6.0,
            )
            if bias is not None:
                linear.bias = torch.nn.Parameter(bias)
        elif bits == 4:
            linear = Linear4bitOnline(
                weight,
                bias,
                quant_type="nf4", #fp4/nf4
            )
        else:
            raise ValueError("quantize only support 4/8 bit")
        return linear

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.W_pack = quant(layer.self_attn.W_pack.weight)
        layer.self_attn.o_proj = quant(layer.self_attn.o_proj.weight)
        layer.mlp.gate_proj = quant(layer.mlp.gate_proj.weight)
        layer.mlp.down_proj = quant(layer.mlp.down_proj.weight)
        layer.mlp.up_proj = quant(layer.mlp.up_proj.weight)
    return model

def init_model_weight_int4(config, model, state_dict):
    #replace Params4bit.cuda with Params4bitCuda
    Params4bit.cuda = Params4bitCuda

    for i in range(config.num_hidden_layers):
        weight_data = state_dict[f'model.layers.{i}.self_attn.W_pack.weight.data']
        weight_quant_state = state_dict[f'model.layers.{i}.self_attn.W_pack.weight.quant_state']
        model.model.layers[i].self_attn.W_pack.weight = Params4bit(weight_data, requires_grad=False, quant_state=weight_quant_state)
        
        weight_data = state_dict[f'model.layers.{i}.self_attn.o_proj.weight.data']
        weight_quant_state = state_dict[f'model.layers.{i}.self_attn.o_proj.weight.quant_state']
        model.model.layers[i].self_attn.o_proj.weight = Params4bit(weight_data, requires_grad=False, quant_state=weight_quant_state)
        
        weight_data = state_dict[f'model.layers.{i}.mlp.gate_proj.weight.data']
        weight_quant_state = state_dict[f'model.layers.{i}.mlp.gate_proj.weight.quant_state']
        model.model.layers[i].mlp.gate_proj.weight = Params4bit(weight_data, requires_grad=False, quant_state=weight_quant_state)
        
        weight_data = state_dict[f'model.layers.{i}.mlp.up_proj.weight.data']
        weight_quant_state = state_dict[f'model.layers.{i}.mlp.up_proj.weight.quant_state']
        model.model.layers[i].mlp.up_proj.weight = Params4bit(weight_data, requires_grad=False, quant_state=weight_quant_state)
        
        weight_data = state_dict[f'model.layers.{i}.mlp.down_proj.weight.data']
        weight_quant_state = state_dict[f'model.layers.{i}.mlp.down_proj.weight.quant_state']
        model.model.layers[i].mlp.down_proj.weight = Params4bit(weight_data, requires_grad=False, quant_state=weight_quant_state)
        
        model.model.layers[i].input_layernorm.weight = state_dict[f'model.layers.{i}.input_layernorm.weight']
        model.model.layers[i].post_attention_layernorm.weight = state_dict[f'model.layers.{i}.post_attention_layernorm.weight']
    
    model.model.embed_tokens.weight = state_dict['model.embed_tokens.weight']
    model.model.norm.weight = state_dict['model.norm.weight']
    model.lm_head.weight = state_dict['lm_head.weight'] 
    return model

def extract_weight_to_half(weight: torch.Tensor, scale_list: torch.Tensor, source_bit_width: int):
    # assert scale_list.dtype in [torch.half, torch.bfloat16]
    assert weight.dtype in [torch.int8]
    if source_bit_width == 8:
        return weight.to(scale_list.dtype) * scale_list[:, None]
    else:
        assert False, "Unsupported bit-width"

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
        layer.self_attn.W_pack = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.self_attn.W_pack.weight,
            bias=layer.self_attn.W_pack.bias,
            dtype=layer.self_attn.W_pack.weight.dtype,
            device=layer.self_attn.W_pack.weight.device if device is None else device,
            empty_init=empty_init
        )
        layer.self_attn.o_proj= QuantizedLinear(
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
