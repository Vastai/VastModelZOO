# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch

from models.ecbsr import ECBSR
from models.plainsr import PlainSR


class Args:
    def __init__(self):
        self.m_ecbsr = 4
        self.c_ecbsr = 8
        self.colors = 1
        self.scale = 2
        self.idt_ecbsr = 0
        self.act_type = 'prelu'


if __name__ == '__main__':


    checkpoint = "experiments/ecbsr-x2-m4c8-prelu-2023-0714-1147/models/model_x2_100.pt"

    device = torch.device('cpu')
    args = Args()

    model_ecbsr = ECBSR(module_nums=args.m_ecbsr, channel_nums=args.c_ecbsr, with_idt=args.idt_ecbsr, act_type=args.act_type, scale=args.scale, colors=args.colors).to(device)
    model_plain = PlainSR(module_nums=args.m_ecbsr, channel_nums=args.c_ecbsr, act_type=args.act_type, scale=args.scale, colors=args.colors).to(device)
    model_ecbsr.load_state_dict(torch.load(checkpoint, map_location="cpu"))

     ## copy weights from ecbsr to plainsr
    depth = len(model_ecbsr.backbone)
    for d in range(depth):
        module = model_ecbsr.backbone[d]
        act_type = module.act_type
        RK, RB = module.rep_params()
        model_plain.backbone[d].conv3x3.weight.data = RK
        model_plain.backbone[d].conv3x3.bias.data = RB

        if act_type == 'relu':     pass
        elif act_type == 'linear': pass
        elif act_type == 'prelu':  model_plain.backbone[d].act.weight.data = module.act.weight.data
        else: raise ValueError('invalid type of activation!')

    # model_ecbsr.eval()
    # model_plain.eval()
    
    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 1, 1080, 1920).to(device)
    # flops, params = profile(model_ecbsr, inputs=(input,))
    # print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    # flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    # print("params:", params)

    flops, params = profile(model_plain, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)

    # experiments/ecbsr-x2-m4c8-prelu-2023-0714-1645/models/model_x2_940.pt (1, 1, 1080, 1920)
    # flops(G): 0.000
    # params: 0.000B
    # flops(G): 12.478
    # params: 2.708K

    input_shape = (1, 1, 1080, 1920) # nchw
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    with torch.no_grad():
        # scripted_model = torch.jit.trace(model_ecbsr, input_data)#.eval()
        # scripted_model.save(checkpoint.replace(".pt", ".torchscript.pt"))

        scripted_model = torch.jit.trace(model_plain, input_data)#.eval()
        scripted_model.save(checkpoint.replace(".pt", "_plain.torchscript.pt"))


    # onnx==10.0.0，opset 10 
    # opset_version=10 can not export onnx in pixel_shuffle
    import onnx
    with torch.no_grad():
        # torch.onnx.export(model_ecbsr, input_data, checkpoint.replace(".pt", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
        # # dynamic_axes= {
        # #             "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
        # #             "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
        # )
        torch.onnx.export(model_plain, input_data, checkpoint.replace(".pt", "_plain.onnx"), input_names=["input"], output_names=["output"], opset_version=11,
        # dynamic_axes= {
        #             "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
        #             "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
        )
    ###############################################################################