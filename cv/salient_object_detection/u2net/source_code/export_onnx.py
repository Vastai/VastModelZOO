# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import torch

from model import U2NET
from model import U2NETP

def main():

    # --------- 1. get image path and name ---------
    model_name = 'u2net'
    # model_name='u2netp'

    model_dir = os.path.join(os.getcwd(), 'saved_models',  model_name + '.pth')
    # model_dir = os.path.join(os.getcwd(), 'saved_models',  'u2net_human_seg' + '.pth')


    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()


    ####################################################################
    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 320, 320)
    flops, params = profile(net, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)
    # u2net_portrait.pth
    # flops(G): 113.113
    # params: 1.131M

    # saved_models/u2net.pth (1, 3, 320, 320)
    # flops(G): 130.610
    # params: 44.010M

    # saved_models/u2netp.pth (1, 3, 320, 320)
    # flops(G): 44.185
    # params: 1.131M

    input_shape = (1, 3, 320, 320)
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(net, input_data).eval()
    scripted_model.save(model_dir.replace(".pth", "-3_320_320.torchscript.pt"))
    # scripted_model = torch.jit.load(model_dir.replace(".pth", "-dynamic.torchscript.pt"))

    import onnx
    torch.onnx.export(net, input_data, model_dir.replace(".pth", "-3_320_320.onnx"), input_names=["input"], output_names=["output"], opset_version=11,
                dynamic_axes= {
                                # "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
                                # "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'},
                                "input": {0: 'batch_size'},
                                "output": {0: 'batch_size'}
                                }
    )

if __name__ == "__main__":
    main()
