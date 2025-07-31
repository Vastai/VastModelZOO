# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch
from net.net import net


if __name__ == '__main__':
    checkpoint = "weights/PairLIE.pth"

    model = net()
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 400, 600)
    flops, params = profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)

    # weights/PairLIE.pth 400, 600
    # flops(G): 182.276
    # params: 341.767K

    input_shape = (1, 3, 400, 600) # nchw
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, input_data)
        scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))

    with torch.no_grad():
        torch.onnx.export(model, input_data, checkpoint.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
        # dynamic_axes= {
        #             "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
        #             "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
        )