# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import ctypes
from enum import Enum
from typing import Union, List
import numpy as np
from space_to_depth_op import vsx, CustomOpBase


class NormalType(Enum):
    NORMAL_EQUAL = 0
    NORMAL_MINUSMEAN_DIVSTD = 1
    NORMAL_DIV255_MINUSMEAN_DIVSTD = 2
    NORMAL_DIV127_5_MINUSONE = 3
    NORMAL_DIV255 = 4


class normalize_para_t(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("in_width_pitch", ctypes.c_uint32),
        ("in_height_pitch", ctypes.c_uint32),
        ("out_width_pitch", ctypes.c_uint32),
        ("out_height_pitch", ctypes.c_uint32),
        ("ch_num", ctypes.c_uint32),
        ("norma_type", ctypes.c_int32),
        ("mean", ctypes.c_uint16 * 4),
        ("std", ctypes.c_uint16 * 4),
    ]


class NormalizeOp(CustomOpBase):
    def __init__(
        self,
        op_name="opf_normalize",
        elf_file="/opt/vastai/vastpipe/data/elf/normalize",
        device_id=0,
        mean=[],
        std=[],
        norm_type=NormalType.NORMAL_EQUAL,
    ):
        super().__init__(op_name=op_name, elf_file=elf_file, device_id=device_id)
        self.mean_ = mean
        self.std_ = std
        self.norm_type_ = norm_type
        if norm_type == NormalType.NORMAL_MINUSMEAN_DIVSTD:
            assert (
                len(mean) == 3
            ), f"len {len(mean)} of mean should be 3 when norm_tye is NORMAL_MINUSMEAN_DIVSTD"
            assert (
                len(std) == 3
            ), f"len {len(std)} of std should be 3 when norm_tye is NORMAL_MINUSMEAN_DIVSTD"
        elif norm_type == NormalType.NORMAL_DIV255_MINUSMEAN_DIVSTD:
            assert (
                len(mean) == 3
            ), f"len {len(mean)} of mean should be 3 when norm_tye is NORMAL_DIV255_MINUSMEAN_DIVSTD"
            assert (
                len(std) == 3
            ), f"len {len(std)} of std should be 3 when norm_tye is NORMAL_DIV255_MINUSMEAN_DIVSTD"

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

            op_conf = normalize_para_t()
            op_conf.width = w
            op_conf.height = h
            op_conf.in_width_pitch = w
            op_conf.in_height_pitch = h
            op_conf.out_width_pitch = w
            op_conf.out_height_pitch = h
            op_conf.ch_num = c
            op_conf.norma_type = self.norm_type_.value
            if (
                self.norm_type_ == NormalType.NORMAL_DIV255_MINUSMEAN_DIVSTD
                or self.norm_type_ == NormalType.NORMAL_MINUSMEAN_DIVSTD
            ):
                op_conf.mean[0] = self.mean_[0]
                op_conf.mean[1] = self.mean_[1]
                op_conf.mean[2] = self.mean_[2]
                op_conf.std[0] = self.std_[0]
                op_conf.std[1] = self.std_[1]
                op_conf.std[2] = self.std_[2]

            op_conf_size = ctypes.sizeof(normalize_para_t)

            outs = self.custom_op_.run_sync(
                tensors=[input],
                config=ctypes.string_at(ctypes.byref(op_conf), op_conf_size),
                output_info=[([c, h, w], vsx.TypeFlag.FLOAT16)],
            )
            outputs.append(outs[0])
        return outputs
