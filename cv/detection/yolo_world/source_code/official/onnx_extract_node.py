# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import onnx
from onnx import numpy_helper
import numpy as np


if __name__ == '__main__':

    npy_save_dir = './onnx_npy_data'
    os.makedirs(npy_save_dir, exist_ok=True)

    # 最终的text模型
    # model = onnx.load('yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6_text-sim_sub.onnx')
    model = onnx.load('./deploy_weights/text_build_run_stream_fp16/yolo_world_clip_backbone.onnx')

    # extract node data
    names_dict = {"clip_text_dict": "baseModel.backbone.text_model.model.text_model.embeddings.token_embedding.weight",
                  "clip_position_dict": "onnx::Add_1099",
                  "layernorm_alpha": "baseModel.backbone.text_model.model.text_model.encoder.layers.0.layer_norm1.weight",
                  "layernorm_beta": "baseModel.backbone.text_model.model.text_model.encoder.layers.0.layer_norm1.bias"}
    reversed_names_dict = {value: key for key, value in names_dict.items()}

    for tensor in model.graph.initializer:
        if tensor.name in names_dict.values():
            weight_np = numpy_helper.to_array(tensor)
            np.save(os.path.join(npy_save_dir, reversed_names_dict[tensor.name]) + '.npy', weight_np)

