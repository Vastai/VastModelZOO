# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from lib.utils.model import get_model
from lib.utils.parser import inference_parser_args as parser_args

def count_op(model, input):
    from thop import profile, clever_format
    flops, params = profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)


def main_test_all(config):

    assert config.resume_checkpoint,"Not find the checkpoint"
    model = get_model(config)
    device = torch.device("cpu")
    flag = 2 if 'stack' in config.resume_checkpoint else 4
    
    model = model.to(device)
    model_checkpoint = torch.load(config.resume_checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(model_checkpoint['state_dict'],strict=True)
    img = torch.randn(1, 3, 256, 256)
    model.eval()
    count_op(model, img)
    torch.onnx.export(model, img, f"hih_wflw_{str(flag)}stack.onnx", input_names = ["input"], opset_version=11)

if __name__ == "__main__":
    config = parser_args()
    main_test_all(config)

# flops(G): 38.205
# params: 18.962M