# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.onnx
import onnx

onnx_path = './decompress_onnx_output/gs_sim.onnx'

def get_sub_model_before():
    node_start = ['y_hat']
    node_end = ['/9/Add_output_0']
    for i in range(len(node_end)):
        onnx_sub_path = onnx_path.replace('.onnx', '_0_58.onnx')
        onnx.utils.extract_model(onnx_path, onnx_sub_path, node_start, node_end)

def get_sub_model_after():
    node_start = ['/9/Add_output_0']
    node_end = ['x_hat']
    for i in range(len(node_end)):
        onnx_sub_path = onnx_path.replace('.onnx', '_58_68.onnx')
        onnx.utils.extract_model(onnx_path, onnx_sub_path, node_start, node_end)


if __name__ == "__main__":
    get_sub_model_before()
    get_sub_model_after()
       

