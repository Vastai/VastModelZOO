# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch
import onnx
from onnxsim import simplify
from modelscope.models import Model

model_id = 'damo/nlp_structbert_sentiment-classification_chinese-base'
model = Model.from_pretrained(model_id)

input_seq_len = 256
input_ids = torch.randint(low=0, high=100, size=(1, input_seq_len), dtype=torch.int32)
atten_mask = torch.ones((1, input_seq_len), dtype=torch.int32)
token_type_ids = torch.randint(low=0, high=1, size=(1, input_seq_len), dtype=torch.int32)
inputs = (input_ids, atten_mask, token_type_ids)

path_to_onnx = "model_ori.onnx"

torch.onnx.export(
    model,
    inputs,
    path_to_onnx,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
)

def simplify_onnx(onnx_model_path, onnxsim_model_path="model_sim_allint32.onnx"):

    onnx_model = onnx.load_model(onnx_model_path, load_external_data=True)
    print(f"load onnx model from {onnx_model_path} to simplify.")
    # use onnx input dict
    shape_dict = {}
    for _, input_i in enumerate(onnx_model.graph.input):
        shape = []
        name = input_i.name
        for _, dim in enumerate(input_i.type.tensor_type.shape.dim):
            shape.append(dim.dim_value)
        shape_dict[name] = shape

    # simplify, delete and merge node
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(
        model_simp,
        onnxsim_model_path
    )
    print("onnx save successed.")

simplify_onnx(path_to_onnx)