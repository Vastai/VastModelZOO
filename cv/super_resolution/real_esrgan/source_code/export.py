# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def main(args):
    # An instance of the model
    # RealESRNet_x4plus
    if args.model == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    # RealESRGAN_x4plus_anime_6B
    elif args.model == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    # RealESRGAN_x2plus
    elif args.model == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    # realesr-animevideov3
    elif args.model == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    # realesr-general-x4v3
    elif args.model == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')

    if args.params:
        keyname = 'params'
    else:
        keyname = 'params_ema'
    keyname = 'params_ema'
    model.load_state_dict(torch.load(args.input, map_location="cpu")[keyname])
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()


    ##############################################################################

    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 128, 128)
    flops, params = profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)
    # realesrgan
    # flops(G): 653.675
    # params: 16.698M

    # realesrnet
    # flops(G): 653.675
    # params: 16.698M

    # RealESRGAN_x4plus_anime_6B
    # flops(G): 208.149
    # params: 4.468M

    # RealESRGAN_x2plus
    # flops(G): 163.466
    # params: 16.703M

    # realesr-animevideov3
    # flops(G): 22.586
    # params: 620.336K
    # RuntimeError: ONNX export failed: Couldn't export operator aten::pixel_shuffle

    # realesr-general-x4v3
    # flops(G): 44.098
    # params: 1.211M
    # RuntimeError: ONNX export failed: Couldn't export operator aten::pixel_shuffle


    input_shape = (1, 3, 128, 128)
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, input_data)#.eval()
        scripted_model.save(args.input.replace(".pth", ".torchscript.pt"))
        scripted_model = torch.jit.load(args.input.replace(".pth", ".torchscript.pt"))

    # onnx==10.0.0，opset 13
    import onnx
    with torch.no_grad():
        torch.onnx.export(model, input_data, args.input.replace(".pth", "_dynamic.onnx"), input_names=["input"], output_names=["output"], opset_version=13,
        dynamic_axes= {
                    "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
                    "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
        )

    ##############################################################################


    # # An example input
    # x = torch.rand(1, 3, 64, 64)
    # # Export the model
    # with torch.no_grad():
    #     torch_out = torch.onnx._export(model, x, args.output, opset_version=11, export_params=True)
    # print(torch_out.shape)


if __name__ == '__main__':
    """Convert pytorch model to onnx models"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=str, default='weights/realesr-general-x4v3.pth', help='Input model path')
    parser.add_argument('--model', type=str, default='realesr-general-x4v3', help='Model name')
    parser.add_argument('--params', action='store_false', help='Use params instead of params_ema')
    args = parser.parse_args()

    main(args)
