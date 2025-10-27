# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import vaststreamx as vsx
import numpy as np
from typing import Union, List
import ctypes
from enum import Enum
attr = vsx.AttrKey

class CustomOpBase:
    def __init__(self, op_name, elf_file, device_id):
        self.device_id_ = device_id
        vsx.set_device(device_id)
        self.custom_op_ = vsx.CustomOperator(op_name=op_name, elf_file_path=elf_file)

class TENSORIZE_FMT(Enum):
    TENSORIZE_FMT_CHW = 0
    TENSORIZE_FMT_HWC = 1


class tensorize_op_t(ctypes.Structure):
    _fields_ = [
        ("src_dim", ctypes.c_uint32 * 3),
        ("src_pitch", ctypes.c_uint32 * 3),
        ("ele_size", ctypes.c_uint32),
        ("src_fmt", ctypes.c_uint32),
    ]


class TensorizeOp(CustomOpBase):
    def __init__(
        self,
        op_name="tensorize_op",
        elf_file="/opt/vastai/vastpipe/data/elf/tensorize_ext_op",
        device_id=0,
    ):
        super().__init__(op_name=op_name, elf_file=elf_file, device_id=device_id)

    def process(
        self, input: Union[List[np.ndarray], List[vsx.Tensor], np.ndarray, vsx.Tensor]
    ):
        if isinstance(input, list):
            if isinstance(input[0], np.ndarray):
                return self.process([vsx.from_numpy(x, self.device_id_) for x in input])
            else:
                return self.process_impl(input)
        return self.process([input])[0]

    def process_impl(self, inputs):
        assert len(inputs) == 1
        outputs = []
        for input in inputs:
            c, h, w = input.shape[-3:]

            op_conf = tensorize_op_t()
            op_conf.src_dim = (c, h, w)
            op_conf.src_pitch = (c, h, w)
            op_conf.ele_size = ctypes.sizeof(ctypes.c_short)
            op_conf.src_fmt = TENSORIZE_FMT.TENSORIZE_FMT_CHW.value

            op_conf_size = ctypes.sizeof(tensorize_op_t)

            outs = self.custom_op_.run_sync(
                tensors=[input],
                config=ctypes.string_at(ctypes.byref(op_conf), op_conf_size),
                output_info=[([c, h, w], vsx.TypeFlag.FLOAT16)],
            )
            print(f"custom_op_ shape:{outs[0].shape}")
            outputs.append(outs[0])
        return outputs

def imagetype_to_vsxformat(imagetype):
    if imagetype == 0:
        return vsx.ImageFormat.YUV_NV12
    elif imagetype == 5000:
        return vsx.ImageFormat.RGB_PLANAR
    elif imagetype == 5001:
        return vsx.ImageFormat.BGR_PLANAR
    elif imagetype == 5002:
        return vsx.ImageFormat.RGB_INTERLEAVE
    elif imagetype == 5003:
        return vsx.ImageFormat.BGR_INTERLEAVE
    elif imagetype == 5004:
        return vsx.ImageFormat.GRAY
    else:
        assert False, f"Unrecognize image type {imagetype}"

def cv_rgb_image_to_vastai(image_cv, device_id):
    assert len(image_cv.shape) >= 2
    h, w = image_cv.shape[:2]
    if len(image_cv.shape) == 3:
        return vsx.create_image(
            image_cv, vsx.ImageFormat.BGR_INTERLEAVE, w, h, device_id
        )
    elif len(image_cv.shape) == 2:
        return vsx.create_image(image_cv, vsx.ImageFormat.GRAY, w, h, device_id)
    else:
        raise Exception("unsupported ndarray shape:", image_cv.shape)

def get_activation_aligned(
    activation, dtype=np.float16, fc_mode=False, force_int8_layout_to_fp16=False
):  # NCHW
    N = C = H = W = 1
    if len(activation.shape) == 2:
        N, C = activation.shape
        fc_mode = True
    elif len(activation.shape) == 5:
        N, C, H, W, B = activation.shape
    elif len(activation.shape) == 1:
        (C,) = activation.shape
    else:
        N, C, H, W = activation.shape
    h_group = w_group = c_group = 0
    if H == 1 and W == 1 and fc_mode == True:
        if dtype == np.float16:
            h_group, w_group, c_group = 1, 1, 256
        elif dtype == np.int8:
            h_group, w_group, c_group = 1, 1, 512
    else:
        if dtype == np.float16 or force_int8_layout_to_fp16:
            h_group, w_group, c_group = 8, 8, 4
        elif dtype == np.int8:
            h_group, w_group, c_group = 8, 8, 8
    pad_H, pad_W, pad_C = H, W, C
    if H % h_group != 0:
        pad_h = h_group - H % h_group
        pad_H += pad_h
    if W % w_group != 0:
        pad_w = w_group - W % w_group
        pad_W += pad_w
    if C % c_group != 0:
        pad_c = c_group - C % c_group
        pad_C += pad_c
    # tensorize to WHC4c8h8w
    w_num = pad_W // w_group
    h_num = pad_H // h_group
    c_num = pad_C // c_group
    n_num = N
    block_size = w_group * h_group * c_group
    activation = activation.astype(dtype)
    np_arr = np.zeros((n_num, w_num, h_num, c_num, block_size), dtype)
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    addr = (
                        (c % c_group) * h_group * w_group
                        + (h % h_group) * w_group
                        + (w % w_group)
                    )
                    if len(activation.shape) == 2:
                        np_arr[n, w // w_group, h // h_group, c // c_group, addr] = (
                            activation[n, c]
                        )
                    elif len(activation.shape) == 1:
                        np_arr[n, w // w_group, h // h_group, c // c_group, addr] = (
                            activation[n]
                        )
                    else:
                        np_arr[n, w // w_group, h // h_group, c // c_group, addr] = (
                            activation[n, c, h, w]
                        )
    return np_arr


class VastaiDynamicGaHa:
    def __init__(
        self,
        module_info,
        vdsp_config,
        max_input_size,
        batch_size=1,
        device_id=0,
        patch=256,
    ):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(module_info)
        self.model_.set_input_shape(max_input_size)
        self.model_.set_batch_size(batch_size)
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
                    return imagetype_to_vsxformat(imagetype)
                else:
                    return vsx.ImageFormat.YUV_NV12
        assert False, "Can't find fusion op that op_type >= 100"

    def process(self, input: Union[np.ndarray, vsx.Image]):
        if isinstance(input, np.ndarray):
            input = cv_rgb_image_to_vastai(input, self.device_id_)
        w, h = input.width, input.height
        p = self.patch_
        w_patch = (w + p - 1) // p * p
        h_patch = (h + p - 1) // p * p
        if w_patch > 1024 or h_patch > 1024:
            print("Warning: image size is too large, dynamic shape max to 1024x1024")
            return None
        top, left = 0, 0
        right = w_patch - w - left
        bottom = h_patch - h - top

        # print(f"w:{w}, h:{h}, new_w:{w_patch}, new_h:{h_patch},bottom:{bottom}, right:{right}")
        params = {
            "rgb_letterbox_ext": [],
            "dynamic_model_input_shapes": [[[1, 3, h_patch, w_patch]]],
        }
        params["rgb_letterbox_ext"].append((w, h, top, bottom, left, right))
        outputs = self.stream_.run_sync([input], params)[0]
        return [vsx.as_numpy(out).astype(np.float32) for out in outputs]


class VastaiDynamicHsChunk:
    def __init__(self, module_info, max_input_size, batch_size=1, device_id=1):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(module_info)
        self.model_.set_input_shape(max_input_size)
        self.model_.set_batch_size(batch_size)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_HOST)
        self.graph_.add_operators(self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

    def process(self, input: np.ndarray):
        aligned_input = get_activation_aligned(input.astype(np.float16))
        vsx_tensor = vsx.from_numpy(aligned_input, self.device_id_)
        (n, c, h, w) = input.shape
        params = {"dynamic_model_input_shapes": [[[n, c, h, w]]]}
        outputs = self.stream_.run_sync([[vsx_tensor]], params)[0]
        return [vsx.as_numpy(out).astype(np.float32) for out in outputs]


class VastaiDynamicGs:
    def __init__(
        self,
        gs0_module_info,
        gs0_max_input_size,
        gs_module_info,
        gs_max_input_size,
        tensorize_elf_path,
        batch_size=1,
        device_id=0,
        gs0_hw_config="",
        gs_hw_config="",
    ):

        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0

        self.gs0_model_ = vsx.Model(gs0_module_info)
        self.gs0_model_.set_input_shape(gs0_max_input_size)
        self.gs0_model_.set_batch_size(batch_size)
        self.gs0_model_op_ = vsx.ModelOperator(self.gs0_model_)
        self.gs0_graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_884_DEVICE)
        self.gs0_graph_.add_operators(self.gs0_model_op_)
        self.gs0_stream_ = vsx.Stream(self.gs0_graph_, vsx.StreamBalanceMode.RUN)
        self.gs0_stream_.register_operator_output(self.gs0_model_op_)
        self.gs0_stream_.build()
        self.tensor_op_ = TensorizeOp(
            elf_file=tensorize_elf_path,
            device_id=device_id,
        )

        self.gs_model_ = vsx.Model(gs_module_info)
        self.gs_model_.set_input_shape(gs_max_input_size)
        self.gs_model_.set_batch_size(batch_size)
        self.gs_model_op_ = vsx.ModelOperator(self.gs_model_)
        self.gs_graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE)
        self.gs_graph_.add_operators(self.gs_model_op_)
        self.gs_stream_ = vsx.Stream(self.gs_graph_, vsx.StreamBalanceMode.RUN)
        self.gs_stream_.register_operator_output(self.gs_model_op_)
        self.gs_stream_.build()

    def process(self, input: np.ndarray):
        # aligned_input = get_activation_aligned(input.astype(np.float16))
        # vsx_tensor = vsx.from_numpy(aligned_input, self.device_id_)
        vsx_tensor = self.tensor_op_.process(input.astype(np.float16))
        (n, c, h, w) = input.shape
        gs0_params = {"dynamic_model_input_shapes": [[[n, c, h, w]]]}
        gs0_outputs = self.gs0_stream_.run_sync([[vsx_tensor]], gs0_params)[0]
        self.gs0_output_shape_ = self.gs0_model_.output_shape
        gs_params = {"dynamic_model_input_shapes": [[gs0_outputs[0].shape]]}
        outputs = self.gs_stream_.run_sync([gs0_outputs], gs_params)[0]

        return [vsx.as_numpy(out).astype(np.float32) for out in outputs]
