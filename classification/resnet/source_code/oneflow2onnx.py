# ==============================================================================
# 
# Copyright (C) 2023 VastaiTech Technologies Inc.  All rights reserved.
# 
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :     vastml
@Time  : 2023/04/07 22:10:00
'''

import argparse
import sys
import os
import onnx
from flowvision.models import ModelCreator
import oneflow as flow
from oneflow import nn
from flowvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check
import glob

def rename_onnx_node(model_path, origin_names, new_names):
    model = onnx.load(model_path)

    # reset args.origin_names
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    origin_names = [in_name, out_name]

    output_tensor_names = set()
    for ipt in model.graph.input:
        output_tensor_names.add(ipt.name)
    for node in model.graph.node:
        for out in node.output:
            output_tensor_names.add(out)

    for origin_name in origin_names:
        if origin_name not in output_tensor_names:
            print("[ERROR] Cannot find tensor name '{}' in onnx model graph.".format(origin_name))
            sys.exit(-1)
    if len(set(origin_names)) < len(origin_names):
        print("[ERROR] There's dumplicate name in --origin_names, which is not allowed.")
        sys.exit(-1)
    if len(new_names) != len(origin_names):
        print("[ERROR] Number of --new_names must be same with the number of --origin_names.")
        sys.exit(-1)
    if len(set(new_names)) < len(new_names):
        print("[ERROR] There's dumplicate name in --new_names, which is not allowed.")
        sys.exit(-1)
    for new_name in new_names:
        if new_name in output_tensor_names:
            print("[WARMING] The defined new_name '{}' is already exist in the onnx model, which is not allowed.")
            #sys.exit(-1)

    for i, ipt in enumerate(model.graph.input):
        if ipt.name in origin_names:
            idx = origin_names.index(ipt.name)
            model.graph.input[i].name = new_names[idx]

    for i, node in enumerate(model.graph.node):
        for j, ipt in enumerate(node.input):
            if ipt in origin_names:
                idx = origin_names.index(ipt)
                model.graph.node[i].input[j] = new_names[idx]
        for j, out in enumerate(node.output):
            if out in origin_names:
                idx = origin_names.index(out)
                model.graph.node[i].output[j] = new_names[idx]

    for i, out in enumerate(model.graph.output):
        if out.name in origin_names:
            idx = origin_names.index(out.name)
            model.graph.output[i].name = new_names[idx]
    
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    print("[Finished] The new model saved in {}.".format(model_path))
    print("[DEBUG INFO] The inputs of new model: {}".format([x.name for x in model.graph.input]))
    print("[DEBUG INFO] The outputs of new model: {}".format([x.name for x in model.graph.output]))


class ResNetGraph(nn.Graph):
    def __init__(self, eager_model):
        super().__init__()
        self.model = eager_model

    def build(self, x):
        return self.model(x)

resnet_model = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
save_dir = './onnx/'
for m in resnet_model:
    model = ModelCreator.create_model(m, pretrained=True)
    
for m in resnet_model:
    MODEL_PARAMS = 'resnet/' + m
    params = flow.load(MODEL_PARAMS)
    model = eval(m)()
    model.load_state_dict(params)

    # 将模型设置为 eval 模式
    model.eval()

    resnet_graph = ResNetGraph(model)
    # 构建出静态图模型
    resnet_graph._compile(flow.randn(1, 3, 224, 224))

    # 导出为 ONNX 模型并进行检查
    convert_to_onnx_and_check(resnet_graph, 
                            onnx_model_path=save_dir + m + '.onnx', 
                            print_outlier=True,
                            dynamic_batch_size=True)
    
onnx_files = glob.glob(save_dir + '/*.onnx')
for onnx_file in onnx_files:
    rename_onnx_node(onnx_file, [' ', ' '], ['input', 'output'])