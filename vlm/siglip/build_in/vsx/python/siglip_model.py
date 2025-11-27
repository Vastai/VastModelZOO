
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import sys

# current_file_path = os.path.dirname(os.path.abspath(__file__))
# common_path = os.path.join(current_file_path, "../..")
# sys.path.append(common_path)

from enum import Enum
import ctypes
import cv2
import argparse
import numpy as np
from typing import Union, List
import vaststreamx as vsx
import copy

class LayoutType(Enum):
    NCHW = 0
    MATRIX_A = 1

class CustomOpBase:
    def __init__(self, op_name, elf_file, device_id):
        self.device_id_ = device_id
        vsx.set_device(device_id)
        self.custom_op_ = vsx.CustomOperator(op_name=op_name, elf_file_path=elf_file)

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

        assert (
            oh_align // 16 * 16 == oh_align
        ), f"oh_align {oh_align} must be aligned to 16"
        assert (
            ow_align // 16 * 16 == ow_align
        ), f"ow_align {ow_align} must be aligned to 16"

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
            assert (
                out_h <= self.oh_align_
            ), f"error: real output height {out_h} must be smaller than oh_align {self.oh_align_}"
            assert (
                out_w <= self.ow_align_
            ), f"error: real output width {out_w} must be smaller than ow_align {self.ow_align_}"

            op_conf = space_to_depth_t()
            op_conf.input_dims_num = 4
            op_conf.input_dims[0] = n
            op_conf.input_dims[1] = c
            op_conf.input_dims[2] = (h // self.kh_)*self.kh_
            op_conf.input_dims[3] = (w // self.kw_)*self.kw_
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
            outs = self.custom_op_.run_sync(
                tensors=[input],
                config=ctypes.string_at(ctypes.byref(op_conf), op_conf_size),
                output_info=[([self.oh_align_, self.ow_align_], vsx.TypeFlag.FLOAT16)],
            )
            outputs.append(outs[0])
        return outputs


def np_softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    res =  e_x / e_x.sum(axis=axis, keepdims=True)
    return res


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
        raise Exception("unsupported ndarray shape", image_cv.shape)


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
        elf_file="/path/to/elf/normalize",
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


class ModelBase:
    def __init__(
        self, model_prefix, vdsp_config, batch_size=1, device_id=0, hw_config="",do_copy=True
    ) -> None:
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.preproc_ops_ = vsx.Operator.load_ops_from_json_file(vdsp_config)
        assert len(self.preproc_ops_) > 0, "load_ops_from_json_file failed"
        self.graph_ = vsx.Graph(do_copy)
        self.graph_.add_operators(*self.preproc_ops_, self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

    @property
    def batch_size(self):
        return self.model_.batch_size

    @property
    def max_batch_size(self):
        return self.model_.max_batch_size

    @property
    def input_count(self):
        return self.model_.input_count

    @property
    def output_count(self):
        return self.model_.output_count

    @property
    def input_shape(self):
        return self.model_.input_shape

    @property
    def output_shape(self):
        return self.model_.output_shape


class ModelCV(ModelBase):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        do_copy=True,
    ) -> None:
        super().__init__(
            model_prefix, vdsp_config, batch_size, device_id, hw_config, do_copy
        )

    def process(
        self, input: Union[List[np.ndarray], List[vsx.Image], np.ndarray, vsx.Image]
    ):
        if isinstance(input, list):
            if isinstance(input[0], np.ndarray):
                return self.process(
                    [cv_rgb_image_to_vastai(x, self.device_id_) for x in input]
                )
            else:
                return self.process_impl(input)
        return self.process([input])[0]

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height = input_shape[-2]
        width = input_shape[-1]
        dummy = np.zeros((height, width, 3), dtype=dtype)
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            vacc_dummy = vsx.create_image(
                dummy, vsx.ImageFormat.BGR_INTERLEAVE, width, height, self.device_id_
            )
            return [vacc_dummy] * batch_size

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        # return [[vsx.as_numpy(o) for o in out] for out in outputs]
        return [vsx.as_numpy(out[0]) for out in outputs]


class SiglipImage:
    def __init__(
        self,
        model_prefix,
        norm_op_elf,
        space2depth_op_elf,
        batch_size=1,
        device_id=0,
        norm_type=NormalType.NORMAL_DIV255,
        hw_config="",
        output_type=vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_HOST,
    ) -> None:
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0

        # normalize op
        mean = np.array([14260, 14163, 13960], dtype=np.uint16)
        std = np.array([13388, 13358, 13418], dtype=np.uint16)
        # norm_type = NormalType.NORMAL_DIV255_MINUSMEAN_DIVSTD
        self.normalize_op_ = NormalizeOp(
            elf_file=norm_op_elf,
            device_id=device_id,
            mean=mean,
            std=std,
            norm_type=norm_type,
        )

        # space_to_depth op
        # kh, kw, out_h, out_w = 32, 32, 64, 4096
            # int kh = 14, kw = 14, oh_align = 256, ow_align = 784;
        kh, kw, out_h, out_w = 14, 14, 736, 784
        self.space_to_depth_op_ = SpaceToDepthOp(
            kh=kh,
            kw=kw,
            oh_align=out_h,
            ow_align=out_w,
            elf_file=space2depth_op_elf,
            device_id=device_id,
        )

        # model
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        # self.preproc_ops_ = vsx.Operator.load_ops_from_json_file("../../data/configs/siglip.json")

        self.graph_ = vsx.Graph(output_type)
        # self.graph_.add_operators(*self.preproc_ops_, self.model_op_)
        self.graph_.add_operators(self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

    @property
    def batch_size(self):
        return self.model_.batch_size

    @property
    def max_batch_size(self):
        return self.model_.max_batch_size

    @property
    def input_count(self):
        return self.model_.input_count

    @property
    def output_count(self):
        return self.model_.output_count

    @property
    def input_shape(self):
        return self.model_.input_shape

    @property
    def output_shape(self):
        return self.model_.output_shape

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height = input_shape[-2]
        width = input_shape[-1]
        dummy = np.zeros((height, width, 3), dtype=dtype)
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            vacc_dummy = vsx.create_image(
                dummy, vsx.ImageFormat.BGR_INTERLEAVE, width, height, self.device_id_
            )
            return [vacc_dummy] * batch_size

    def process(
        self, input: Union[List[np.ndarray], List[vsx.Image],List[vsx.Tensor], np.ndarray, vsx.Image, vsx.Tensor]
    ):
        if isinstance(input, list):
            if isinstance(input[0], np.ndarray):
                return self.process(
                    [cv_rgb_image_to_vastai(x, self.device_id_) for x in input]
                )
                # input_tensors = []
                # for image in input:
                #     mod_h, mod_w = self.model_.input_shape[0][-2:]
                #     oh, ow = mod_h, mod_w
                #     print(f"input shape:{image.shape}")
                #     np.save('input_img.npy', image)
                #     img = self.letterbox(image, (mod_w, mod_h))[0]
                #     np.save('letterbox.npy', img)
                #     img = img[:, :, ::-1].transpose(2, 0, 1)
                #     img = np.ascontiguousarray(img)
                #     img = torch.from_numpy(img).to('cpu')
                #     img = img.float()  # uint8 to fp16/32
                #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
                #     if img.ndimension() == 3:
                #         img = img.unsqueeze(0)
                #     img = img.numpy()
                #     np.save('before.npy', img)
                #     input_tensors.append(vsx.from_numpy(img))
                # return self.process(input_tensors)
            else:
                return self.process_impl(input)
        return self.process([input])[0]

    def compute_size(self, img_w, img_h, size):
        if isinstance(size, int):
            size_h, size_w = size, size
        elif len(size) < 2:
            size_h, size_w = size[0], size[0]
        else:
            size_h, size_w = size[-2:]

        # r = max(size_w / img_w, size_h / img_h)
        r = min(size_w / img_w, size_h / img_h)
        r = min(r, 1.0)

        new_w = int(round(r * img_w))
        new_h = int(round(r * img_h))
        # print(f"new_w:{new_w}, new_h:{new_h}")
        return (new_w, new_h)
    def get_aligned(self, activate):  # NCHW or HW
        # if isinstance(activate, torch.Tensor):
        #     activate = activate.numpy()
        if isinstance(activate, vsx.Tensor):
            activate = vsx.as_numpy(activate)
        if ( (len(activate.shape) == 3 and activate.shape[0] > 1 and activate.shape[1] > 1 and activate.shape[2] > 1) or
            (len(activate.shape) == 4 and activate.shape[0] == 1 and activate.shape[1] > 1 and activate.shape[2] > 1 and activate.shape[3] > 1)):  # tensor 三维或者四维 884 对齐，但是不转
            if len(activate.shape) == 3:
                activate = activate.reshape(1, activate.shape[0], activate.shape[1], activate.shape[2])
            n, channel, height, width = activate.shape
            # 如果通道数不是 4 的倍数，则用 0 填充通道数至 4 的倍数
            if channel % 4 != 0 or height % 8 != 0 or width %8 != 0 :
                pad_c, pad_h, pad_w = 0, 0, 0
                if channel % 4 != 0:
                    pad_c = 4 - channel % 4
                if height % 8 != 0:
                    pad_h = 8 - height % 8
                if width % 8 != 0:
                    pad_w = 8 - width % 8
                activate = np.pad(activate, ((0,0),(0,pad_c), (0,pad_h), (0,pad_w)), mode='constant')
        elif(len(activate.shape) == 2 and activate.shape[0] > 1 and activate.shape[1] > 1): # 矩阵 16 对齐
            height, width = activate.shape
            # 如果通道数不是 16 的倍数，则用 0 填充通道数至 16 的倍数
            align_factor = 16
            if height % align_factor != 0 or width % align_factor != 0 :
                pad_h, pad_w = 0, 0
                if height % align_factor != 0:
                    pad_h = align_factor - height % align_factor
                if width % align_factor != 0:
                    pad_w = align_factor - width % align_factor
                activate = np.pad(activate, ((0, pad_h), (0, pad_w)), mode='constant')
        else:
            assert False, "activate shape error: {}".format(activate.shape)
        return activate
                  
    def get_matrixA(self, activate):
        # if isinstance(activate, torch.Tensor):
        #     activate = activate.numpy()
        if isinstance(activate, vsx.Tensor):
            activate = vsx.as_numpy(activate)
        if activate.dtype == np.float32:
            activate = activate.astype(np.float16)
        assert len(activate.shape) == 2, "activate shape error: {}".format(activate.shape)
        
        height, width = activate.shape
        # 如果通道数不是 16 的倍数，则用 0 填充通道数至 16 的倍数
        align_factor = 16
        if height % align_factor != 0 or width % align_factor != 0 :
            pad_h, pad_w = 0, 0
            if height % align_factor != 0:
                pad_h = align_factor - height % align_factor
            if width % align_factor != 0:
                pad_w = align_factor - width % align_factor
            activate = np.pad(activate, ((0, pad_h), (0, pad_w)), mode='constant')
            
        M, K = activate.shape
        activate = activate.reshape(M // 16, 16, K // 16, 16)  # M 16m K 16k 
        activate = activate.transpose(0,2,1,3)                 # M K 16m 16k
        activate = activate.reshape(M // 16 , K // 16, 256)    # M K 256
        return activate
    
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        # print(f"new_unpad:{new_unpad}")
    
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        # print(f"dw, dh:{dw, dh}")
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        # print(f"dw, dh:{dw, dh}, new_unpad:{new_unpad}")
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        # print(f"top, bottom, left, right:{top, bottom, left, right}")
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)
    
    def space_to_depth(self, x, block_size, stride):
        """
            自定义的 space_to_depth 操作，结合 stride 参数。
            
            参数:
            x: 输入的 4D 数组，形状为 (batch_size, channels, height, width)
            block_size: 块大小，表示空间维度被分割的块的大小
            stride: 步长，表示在空间维度上移动的步长
            
            返回:
            转换后的 4D 数组，形状为 (batch_size, channels*block_size*block_size, height_out, width_out)
        """
        batch_size, channels, height, width = x.shape
        assert batch_size == 1, "batch_size must be 1"
        
        height_out = (height - block_size) // stride + 1
        width_out = (width - block_size) // stride + 1
        
        # 初始化输出张量
        output_shape = (batch_size, channels * block_size * block_size, height_out, width_out)
        output = np.zeros(output_shape, dtype=x.dtype)
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(height_out):
                    for w in range(width_out):
                        h_start = h * stride
                        h_end = h_start + block_size
                        w_start = w * stride
                        w_end = w_start + block_size
                        block = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c * block_size * block_size:(c + 1) * block_size * block_size, h, w] = block.flatten()
        
        output = output.reshape((-1, height_out * width_out))   
        return output.transpose()
       
    def process_impl(self, inputs):
        mod_h, mod_w = self.model_.input_shape[0][-2:]
        outputs = []
        for input in inputs:
            w, h = self.compute_size(input.width, input.height, [mod_h, mod_w])
            new_unpad = (w,h)
            dw, dh = (mod_w - w) / 2, (mod_h - h) / 2
            # print(f"dw, dh:{dw, dh}")
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            # print(f"w:{w}, h:{h}")

            # print(f"top, bottom, left, right:{top, bottom, left, right}")
            resize_out = vsx.resize_copy_make_border(input,vsx.ImageResizeType.BILINEAR_CV, mod_w, mod_h, 
                                        vsx.ImagePaddingType.PADDING_TYPE_CONSTANT,
                                        (114,114,114),padding_edges=(top, bottom, left, right))
            
            cvtcolor_out = vsx.cvtcolor(resize_out, vsx.ImageFormat.RGB_PLANAR)

            norm_out = self.normalize_op_.process(cvtcolor_out)
            space_to_depth_out2 = self.space_to_depth_op_.process(norm_out)

            model_outs = self.stream_.run_sync([[space_to_depth_out2]])
            outs = [vsx.as_numpy(out[0]) for out in model_outs]
            outputs.append(outs)
        return outputs
