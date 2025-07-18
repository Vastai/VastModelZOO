import ctypes
from enum import Enum
from typing import Union, List
import numpy as np
import vaststreamx as vsx


class CustomOpBase:
    def __init__(self, op_name, elf_file, device_id):
        self.device_id_ = device_id
        vsx.set_device(device_id)
        self.custom_op_ = vsx.CustomOperator(op_name=op_name, elf_file_path=elf_file)

class LayoutType(Enum):
    NCHW = 0
    MATRIX_A = 1


class DataType(Enum):
    DATA_TYPE_FP16 = 0
    DATA_TYPE_INT8 = 1
    DATA_TYPE_BFP16 = 2
    DATA_TYPE_FLOAT32 = 3
    DATA_TYPE_DOUBLE = 4
    DATA_TYPE_INT32 = 5
    DATA_TYPE_RESERVED = 6


class space_to_depth_t(ctypes.Structure):
    _fields_ = [
        ("input_dims_num", ctypes.c_uint32),
        ("input_dims", ctypes.c_uint32 * 4),
        ("input_align_dims", ctypes.c_uint32 * 4),
        ("input_layout", ctypes.c_int32),
        ("input_type", ctypes.c_int32),
        ("kh", ctypes.c_uint32),
        ("kw", ctypes.c_uint32),
        ("output_dims_num", ctypes.c_uint32),
        ("output_dims", ctypes.c_uint32 * 4),
        ("output_align_dims", ctypes.c_uint32 * 4),
        ("output_layout", ctypes.c_int32),
    ]


class SpaceToDepthOp(CustomOpBase):
    def __init__(
        self,
        kh,
        kw,
        oh_align,
        ow_align,
        op_name="opf_space_to_depth_out_matrix",
        elf_file="/opt/vastai/vastpipe/data/elf/space_to_depth",
        device_id=0,
    ):
        super().__init__(op_name=op_name, elf_file=elf_file, device_id=device_id)

        assert (oh_align//16*16==oh_align),f"oh_align {oh_align} must be aligned to 16"
        assert (ow_align//16*16==ow_align),f"ow_align {ow_align} must be aligned to 16"

        self.kh_ = kh
        self.kw_ = kw
        self.oh_align_ = oh_align
        self.ow_align_ = ow_align

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
        outputs = []
        for input in inputs:
            assert len(input.shape) == 3 or len(input.shape) == 4
            if len(input.shape) == 3:
                n = 1
                c, h, w = input.shape
            else:
                n, c, h, w = input.shape

            out_h = (h // self.kh_) * (w // self.kw_)
            out_w = c * self.kh_ * self.kw_
            assert out_h <= self.oh_align_, f"error: real output height {out_h} must be smaller than oh_align {self.oh_align_}"
            assert out_w <= self.ow_align_, f"error: real output width {out_w} must be smaller than ow_align {self.ow_align_}"

            op_conf = space_to_depth_t()
            op_conf.input_dims_num = 4
            op_conf.input_dims[0] = n
            op_conf.input_dims[1] = c
            op_conf.input_dims[2] = h
            op_conf.input_dims[3] = w
            op_conf.input_align_dims[0] = n
            op_conf.input_align_dims[1] = c
            op_conf.input_align_dims[2] = h
            op_conf.input_align_dims[3] = w
            op_conf.input_layout = LayoutType.NCHW.value
            op_conf.input_type = DataType.DATA_TYPE_FP16.value
            op_conf.kh = self.kh_
            op_conf.kw = self.kw_
            op_conf.output_dims_num = 2
            op_conf.output_dims[0] = out_h
            op_conf.output_dims[1] = out_w
            op_conf.output_align_dims[0] = self.oh_align_
            op_conf.output_align_dims[1] = self.ow_align_
            op_conf.output_layout = LayoutType.MATRIX_A.value

            op_conf_size = ctypes.sizeof(space_to_depth_t)
            self.custom_op_.set_config(
                config_bytes=ctypes.string_at(ctypes.byref(op_conf), op_conf_size),
                config_size=op_conf_size,
            )
            outs = self.custom_op_.execute(
                tensors=[input],
                output_info=[([self.oh_align_, self.ow_align_], vsx.TypeFlag.FLOAT16)],
            )
            outputs.append(outs[0])
        return outputs
