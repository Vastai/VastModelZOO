# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import torch

from basicsr.archs.ecbsr_arch import ECBSR


if __name__ == "__main__":


    model = ECBSR(num_in_ch=3,
                 num_out_ch=3,
                 num_block=2,
                 num_channel=16,
                 with_idt=False,
                 act_type="prelu",
                 scale=2)

    model_path = "experiments/100_train_ECBSR_x2_m4c16_prelu_RGB_mini/models/net_g_1476800.pth"


    device = torch.device("cpu")

    state_dict = torch.load(model_path)['params']
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 1080, 1920)
    flops, params = profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)


    # experiments/100_train_ECBSR_x4_m4c16_prelu_RGB/models/net_g_1078400.pth (1, 3, 1080, 1920) 2x
    # flops(G): 0.369
    # params: 80.000B

    # experiments/100_train_ECBSR_x2_m4c16_prelu_RGB_mini/models/net_g_1476800.pth (1, 3, 1080, 1920) 2x
    # flops(G): 0.221
    # params: 48.000B

    checkpoint = model_path

    input_shape = (1, 3, 1080, 1920) # nchw
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, input_data)#.eval()
        scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))

    # onnx==10.0.0，opset 11
    # import onnx
    with torch.no_grad():
        torch.onnx.export(model, input_data, checkpoint.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
        # dynamic_axes= {
        #             "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
        #             "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
        )
