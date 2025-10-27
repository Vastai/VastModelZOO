# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import vaststreamx as vsx
import numpy as np
import common.utils as utils
import torch
from typing import List
from common.tensorize_op import TensorizeOp

attr = vsx.AttrKey


class VastaiGaHa:
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        patch=64,
    ):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.input_shape_ = self.model_.input_shape[0]
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.preproc_ops_ = vsx.Operator.load_ops_from_json_file(vdsp_config)
        assert len(self.preproc_ops_) > 0, "load_ops_from_json_file failed"
        self.graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE)
        self.graph_.add_operators(*self.preproc_ops_, self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

        self.patch_ = patch

    def get_fusion_op_iimage_format(self):
        for op in self.preproc_ops_:
            if op.op_type >= 100:
                buildin_op = op.cast_to_buildin_operator()
                if "IIMAGE_FORMAT" in list(buildin_op.attributes.keys()):
                    imagetype = buildin_op.get_attribute(attr.IIMAGE_FORMAT)
                    return utils.imagetype_to_vsxformat(imagetype)
                else:
                    return vsx.ImageFormat.YUV_NV12
        assert False, "Can't find fusion op that op_type >= 100"

    def process(self, input: vsx.Image, params: dict = {}):
        outputs = self.stream_.run_sync([input], params)[0]
        return [vsx.as_numpy(out).astype(np.float32) for out in outputs]


class VastaiGs:
    def __init__(
        self, model_prefix, tensorize_elf_path, batch_size=1, device_id=0, hw_config=""
    ):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0

        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE)
        self.graph_.add_operators(self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()
        self.tensorize_elf_path = tensorize_elf_path
        if tensorize_elf_path != "":
            self.tensor_op_ = TensorizeOp(
                elf_file=tensorize_elf_path,
                device_id=device_id,
            )

    def process(self, input):
        vsx_tensor = None
        if self.tensorize_elf_path != "":
            vsx_tensor = self.tensor_op_.process(input.astype(np.float16))
        else:
            aligned_input = utils.get_activation_aligned_faster_1(
                input.astype(np.float16)
            )
            vsx_tensor = vsx.from_numpy(aligned_input, self.device_id_)

        outputs = self.stream_.run_sync([[vsx_tensor]])[0]
        return [vsx.as_numpy(out).astype(np.float32) for out in outputs]


class VastaiHs:
    def __init__(self, model_prefix, batch_size=1, device_id=1, hw_config=""):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_HOST)
        self.graph_.add_operators(self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

    def process(self, input: np.ndarray):
        aligned_input = utils.get_activation_aligned_faster_1(input.astype(np.float16))
        vsx_tensor = vsx.from_numpy(aligned_input, self.device_id_)
        outputs = self.stream_.run_sync([[vsx_tensor]])[0]
        return [vsx.as_numpy(out).astype(np.float32) for out in outputs]
