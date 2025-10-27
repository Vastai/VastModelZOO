# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import onnx
from onnx import helper, shape_inference
import onnxruntime as ort
from scipy import spatial
import numpy as np
import torch


def get_ga_ha(path_onnx_1, path_onnx_2, model_name):
    # 加载两个 ONNX 模型
    model1 = onnx.load(path_onnx_1)
    model2 = onnx.load(path_onnx_2)

    # 获取 model1 的输出节点名称
    model1_output_name = model1.graph.output[0].name
    # 获取 model2 的输入节点名称
    model2_input_name = model2.graph.input[0].name

    # 创建一个新的图
    combined_graph = helper.make_graph(
        nodes=[],  # 先创建一个空的节点列表
        name=model_name,
        inputs=[],  # 先创建一个空的输入列表
        outputs=[],  # 先创建一个空的输出列表
        initializer=[]  # 先创建一个空的初始化器列表
    )

    # 合并 model1 和 model2 的节点
    combined_graph.node.extend(model1.graph.node)
    combined_graph.node.extend(model2.graph.node)

    # 合并 model1 和 model2 的输入
    combined_graph.input.extend(model1.graph.input)

    # 合并 model1 和 model2 的输出
    combined_graph.output.extend(model1.graph.output)
    combined_graph.output.extend(model2.graph.output)

    # 合并 model1 和 model2 的初始化器
    combined_graph.initializer.extend(model1.graph.initializer)
    combined_graph.initializer.extend(model2.graph.initializer)

    # 创建一个新的模型
    combined_model = helper.make_model(combined_graph)
    # 重命名 model2 的输入节点以匹配 model1 的输出节点
    for node in combined_model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name == model2_input_name:
                node.input[i] = model1_output_name

    # 保存合并后的模型
    onnx.save(combined_model, f"{model_name}.onnx")

    # 验证合并后的模型
    inferred_model = shape_inference.infer_shapes(combined_model)
    onnx.save(inferred_model, f"{model_name}.onnx")
    
    
# 通过 cos 相似度比较
def get_cosine(res_before, res_after, thresh_hold=1e-8):
    print('res_before: {f}'.format(f=res_before.shape))
    print('res_after: {f}'.format(f=res_after.shape))
    if res_after is not None and res_before is not None:
        res_before = res_before.flatten().astype("float32")
        res_after = res_after.flatten().astype("float32")
        cos_sim_scipy =  1 - spatial.distance.cosine(res_before, res_after)
        print('cos_sim:' + str(cos_sim_scipy))
        thresh_hold = thresh_hold
        print(res_before.shape)
        print(res_after.shape)
        try:
            np.testing.assert_allclose(res_before, res_after, atol=thresh_hold, rtol=thresh_hold)
            # return True
        except AssertionError as e:
            print(e)
    else:
        print('res_before or res_before is None!')
        print('res_before: {f}'.format(f=res_before))
        print('res_after: {f}'.format(f=res_after))
            
    
def get_onnx_res_dict(path_onnx, data_input): 
    if isinstance(data_input[0], torch.Tensor):    
        data_input = [e.cpu().numpy() for e in data_input]   
    session = ort.InferenceSession(path_onnx)
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]

    input_feed = dict(zip(input_names, data_input))
    result = session.run(output_names, input_feed)
    return dict(zip(output_names, result)) 
    
if __name__ == "__main__":
    path_onnx_1 = './onnx_models/compress/compress_ga_sim.onnx'
    path_onnx_2 = './onnx_models/compress/compress_ha_sim.onnx'
    model_name = 'compress_ga_ha_sim'
    get_ga_ha(path_onnx_1, path_onnx_2, model_name)
    
    # x = np.load('./projects/mlic/scripts/data_res/x_np.npy')
    # y = np.load('./projects/mlic/scripts/data_res/y_np.npy')
    # z = np.load('./projects/mlic/scripts/data_res/z_np.npy')
    # # res_y_z = get_onnx_res_dict(model_name + '.onnx', [x])
    # res_y_z = get_onnx_res_dict('./projects/mlic/scripts/compress_ga_ha_sim_512_768.onnx', [x])
    # get_cosine(y, res_y_z['y'], 1e-6)
    # get_cosine(z, res_y_z['z'], 1e-6)