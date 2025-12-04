# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys
from typing import Union, List
from custom_op_base import CustomOpBase, vsx
import numpy as np
import ctypes


MAX_PP_MODEL_NUM = 10
PP_MODEL_OUTPUT_SIZE = 1024
OUTPUT_COUNT = 500

pointpillar_op_name = "custom_op_pointpillar"


class pointpillar_model_ext_op_t(ctypes.Structure):
    _fields_ = [
        ("num_points", ctypes.c_int32),
        ("num_features", ctypes.c_int32),
        ("voxel_size", ctypes.c_float * 3),
        ("coors_range", ctypes.c_float * 6),
        ("feature_width", ctypes.c_int32),
        ("feature_height", ctypes.c_int32),
        ("pts_per_voxel_max_num", ctypes.c_int32),
        ("shuffle_enabled", ctypes.c_int32),
        ("normalize_enabled", ctypes.c_int32),
        ("tmp_buffer_size", ctypes.c_int32),
        ("tmp_buffer_ptr64", ctypes.c_int64),
        ("valid_model_num", ctypes.c_int32),
        ("max_voxel_num", ctypes.c_int32 * MAX_PP_MODEL_NUM),
        ("model_addr_list", ctypes.c_int64 * MAX_PP_MODEL_NUM),
        ("score_ptr64", ctypes.c_int64),
        ("label_ptr64", ctypes.c_int64),
        ("box_ptr64", ctypes.c_int64),
        ("place_holder", ctypes.c_int64 * 8),
    ]


class Detection3D(CustomOpBase):
    def __init__(
        self,
        model_configs,
        elf_file,
        voxel_sizes,
        coors_range,
        device_id=0,
        max_points_num=120000,
        num_feature=4,
        pts_per_voxel_max_num=32,
        shuffle_enabled=0,
        max_feature_width=864,
        max_feature_height=496,
        actual_feature_width=480,
        actual_feature_height=480,
        normalize_enabled=0,
    ):
        assert len(voxel_sizes) == 3
        assert len(coors_range) == 6

        super().__init__(
            op_name=pointpillar_op_name, elf_file=elf_file, device_id=device_id
        )
        self.custom_op_.set_callback_info(
            [(1, 3, 720, 1280)],
            [
                (1, 3, 720, 1280),
                (1, 3, 540, 960),
                (1, 3, 360, 512),
                (1, 3, 1080, 1920),
            ],
        )
        batch_size = 1
        self.op_conf_ = pointpillar_model_ext_op_t()
        self.op_conf_.valid_model_num = len(model_configs)
        self.models_ = []
        for i in range(len(model_configs)):
            model = vsx.Model(
                model_configs[i].model_prefix, batch_size, model_configs[i].hw_config
            )
            self.op_conf_.max_voxel_num[i] = model_configs[i].max_voxel_num
            self.op_conf_.model_addr_list[i] = model.address
            self.models_.append(model)

        self.op_conf_.num_features = num_feature
        self.op_conf_.voxel_size[0] = voxel_sizes[0]
        self.op_conf_.voxel_size[1] = voxel_sizes[1]
        self.op_conf_.voxel_size[2] = voxel_sizes[2]

        self.op_conf_.coors_range[0] = coors_range[0]
        self.op_conf_.coors_range[1] = coors_range[1]
        self.op_conf_.coors_range[2] = coors_range[2]
        self.op_conf_.coors_range[3] = coors_range[3]
        self.op_conf_.coors_range[4] = coors_range[4]
        self.op_conf_.coors_range[5] = coors_range[5]

        self.op_conf_.feature_width = actual_feature_width
        self.op_conf_.feature_height = actual_feature_height
        self.op_conf_.pts_per_voxel_max_num = pts_per_voxel_max_num
        self.op_conf_.shuffle_enabled = shuffle_enabled
        self.op_conf_.normalize_enabled = normalize_enabled

        fix_buffer_size = (
            max_points_num * 16
            + max_feature_height * max_feature_width * 2
            + model_configs[0].max_voxel_num * 7
            + pts_per_voxel_max_num * model_configs[0].max_voxel_num * 24
            + 21 * 1024 * 1024
        )
        
        np_array = np.random.randint(
            low=0, high=256, size=(fix_buffer_size), dtype=np.uint8
        )
        self.tmp_buffer_tensor_ = vsx.from_numpy(np_array, self.device_id_)
        np_array = np.random.randint(
            low=0, high=256, size=(PP_MODEL_OUTPUT_SIZE), dtype=np.uint8
        )
        self.tmp_out_tensor_ = vsx.from_numpy(np_array, self.device_id_)

        np_array = np.random.rand(OUTPUT_COUNT).astype(np.float16)
        self.score_tensor_ = vsx.from_numpy(np_array, self.device_id_)

        np_array = np.random.rand(OUTPUT_COUNT).astype(np.float16)
        self.label_tensor_ = vsx.from_numpy(np_array, self.device_id_)

        np_array = np.random.rand(OUTPUT_COUNT, 7).astype(np.float16)
        self.box_tensor_ = vsx.from_numpy(np_array, self.device_id_)

        self.op_conf_.tmp_buffer_size = fix_buffer_size
        self.op_conf_.tmp_buffer_ptr64 = self.tmp_buffer_tensor_.addr
        self.op_conf_.score_ptr64 = self.label_tensor_.addr
        self.op_conf_.label_ptr64 = self.score_tensor_.addr
        self.op_conf_.box_ptr64 = self.box_tensor_.addr

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        if ctx == "CPU":
            return [np.zeros(input_shape, dtype=dtype)] * batch_size
        else:
            return [
                vsx.from_numpy(np.zeros(input_shape, dtype=dtype), self.device_id_)
            ] * batch_size

    def process(
        self,
        input: Union[
            List[List[np.ndarray]],
            List[List[vsx.Tensor]],
            List[np.ndarray],
            List[vsx.Tensor],
        ],
    ):
        if isinstance(input, list):
            if isinstance(input[0], np.ndarray):
                return self.process([vsx.from_numpy(x, self.device_id_) for x in input])
            else:
                return self.process_impl(input)
        else:
            return self.process([input])[0]

    def process_impl(self, inputs):
        results = []
        for input in inputs:
            self.op_conf_.num_points = input.size // 4
            op_conf_size = ctypes.sizeof(pointpillar_model_ext_op_t)
            self.custom_op_.run_sync(
                tensors=[input],
                config=ctypes.string_at(ctypes.byref(self.op_conf_), op_conf_size),
                output_info=[
                    ([PP_MODEL_OUTPUT_SIZE], vsx.TypeFlag.UINT8),
                    ([OUTPUT_COUNT], vsx.TypeFlag.FLOAT16),
                    ([OUTPUT_COUNT], vsx.TypeFlag.FLOAT16),
                    ([OUTPUT_COUNT, 7], vsx.TypeFlag.FLOAT16),
                ],
            )
            outs = [
                vsx.as_numpy(self.score_tensor_),
                vsx.as_numpy(self.label_tensor_),
                vsx.as_numpy(self.box_tensor_),
            ]

            results.append(outs)
        return results
