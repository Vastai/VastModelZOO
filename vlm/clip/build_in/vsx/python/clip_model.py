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
from space_to_depth_op import vsx, SpaceToDepthOp, CustomOpBase
import clip
import copy


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


class ClipImage:
    def __init__(
        self,
        model_prefix,
        norm_op_elf,
        space2depth_op_elf,
        batch_size=1,
        device_id=0,
        hw_config="",
        do_copy=True,
    ) -> None:
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0

        # normalize op
        mean = np.array([14260, 14163, 13960], dtype=np.uint16)
        std = np.array([13388, 13358, 13418], dtype=np.uint16)
        norm_type = NormalType.NORMAL_DIV255_MINUSMEAN_DIVSTD
        self.normalize_op_ = NormalizeOp(
            elf_file=norm_op_elf,
            device_id=device_id,
            mean=mean,
            std=std,
            norm_type=norm_type,
        )

        # space_to_depth op
        kh, kw, out_h, out_w = 32, 32, 64, 4096
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
        self.graph_ = vsx.Graph(do_copy)
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

    def compute_size(self, img_w, img_h, size):
        if isinstance(size, int):
            size_h, size_w = size, size
        elif len(size) < 2:
            size_h, size_w = size[0], size[0]
        else:
            size_h, size_w = size[-2:]

        r = max(size_w / img_w, size_h / img_h)

        new_w = round(r * img_w)
        new_h = round(r * img_h)
        return (new_w, new_h)

    def process_impl(self, inputs):
        mod_h, mod_w = self.model_.input_shape[0][-2:]
        outputs = []
        for input in inputs:
            w, h = self.compute_size(input.width, input.height, [mod_h, mod_w])
            cvtcolor_out = vsx.cvtcolor(input, vsx.ImageFormat.RGB_PLANAR)
            resize_out = vsx.resize(
                cvtcolor_out,
                vsx.ImageResizeType.BICUBIC_PILLOW,
                resize_width=w,
                resize_height=h,
            )

            left, top = (w - mod_w) // 2, (h - mod_h) // 2

            crop_out = vsx.crop(resize_out, (left, top, mod_w, mod_h))
            norm_out = self.normalize_op_.process(crop_out)
            space_to_depth_out = self.space_to_depth_op_.process(norm_out)

            model_outs = self.stream_.run_sync([[space_to_depth_out]])

            outs = [vsx.as_numpy(out[0]) for out in model_outs]
            outputs.append(outs)
        return outputs

class ClipText(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        do_copy=True,
    ):
        super().__init__(
            model_prefix=model_prefix,
            vdsp_config=vdsp_config,
            batch_size=batch_size,
            device_id=device_id,
            hw_config=hw_config,
            do_copy=do_copy,
        )

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        tokens = self.make_tokens("test string")
        if ctx == "CPU":
            return [tokens] * batch_size
        else:
            return [
                [vsx.from_numpy(token, self.device_id_) for token in tokens]
            ] * batch_size

    def make_tokens(self, text):
        assert isinstance(text, str), f"input type must be str"
        token = clip.tokenize(text)[0]
        token_padding = np.pad(token.numpy(), pad_width=(0, 3)).astype(np.int32)
        # make mask
        index = np.argmax(token_padding)
        token_mask = copy.deepcopy(token_padding)
        token_mask[: index + 1] = 1
        # make input
        zero_arr = np.zeros(token_padding.shape, dtype=np.int32)
        tokens = []
        tokens.append(token_padding)
        tokens.append(zero_arr)
        tokens.append(zero_arr)
        tokens.append(token_mask)
        tokens.append(zero_arr)
        tokens.append(zero_arr)

        return tokens

    def process(
        self,
        input: Union[
            List[List[vsx.Tensor]],
            List[List[np.ndarray]],
            List[vsx.Tensor],
            List[np.ndarray],
            List[str],
            str,
        ],
    ):
        if isinstance(input, list):
            if isinstance(input[0], list):
                if isinstance(input[0][0], np.ndarray):
                    return self.process(
                        [
                            [
                                vsx.from_numpy(
                                    np.array(x, dtype=np.int32), self.device_id_
                                )
                                for x in one
                            ]
                            for one in input
                        ]
                    )
                else:
                    return self.process_impl(input)
            elif isinstance(input[0], str):
                return self.process([self.make_tokens(x) for x in input])
            elif isinstance(input[0], np.ndarray):
                tensors = [
                    vsx.from_numpy(np.array(x, dtype=np.int32), self.device_id_)
                    for x in input
                ]
                return self.process_impl([tensors])[0]
            else:
                return self.process_impl([input])[0]
        else:
            tokens = self.make_tokens(input)
            return self.process(tokens)

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [vsx.as_numpy(out[0]).astype(np.float32) for out in outputs]


class ClipModel:
    def __init__(
        self,
        imgmod_prefix,
        norm_elf,
        space2depth_elf,
        txtmod_prefix,
        txtmod_vdsp_config,
        batch_size=1,
        device_id=0,
        imgmod_hw_config="",
        txtmod_hw_config="",
    ) -> None:
        self.imgmod_ = ClipImage(
            imgmod_prefix,
            norm_elf,
            space2depth_elf,
            batch_size,
            device_id,
            imgmod_hw_config,
        )
        self.txtmod_ = ClipText(
            txtmod_prefix, txtmod_vdsp_config, batch_size, device_id, txtmod_hw_config
        )
        self.device_id_ = device_id

    def process(self, image, texts):
        img_feature = self.process_image(image)
        txt_features = self.process_texts(texts)

        return self.post_process(img_feature,txt_features)

    def process_image(self, image):
        return self.imgmod_.process(image)

    def process_texts(self, texts):
        return self.txtmod_.process(texts)

    def post_process(self, img_feature, txt_features):
        img_feat = np.multiply(img_feature, 100.00000762939453)
        txt_feat = np.concatenate(txt_features, axis=0)
        txt_feat = np.transpose(txt_feat)
        feature = np.matmul(img_feat, txt_feat).astype("float32")
        scores = np_softmax(feature).squeeze()
        return scores

    def compute_tokens(self, text):
        assert isinstance(text, str), f"input text must be str"
        return self.txtmod_.make_tokens(text=text)


# def argument_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--imgmod_prefix",
#         default="/opt/vastai/vastpipe/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod",
#         help="image model prefix of the model suite files",
#     )
#     parser.add_argument(
#         "--imgmod_hw_config",
#         help="image model hw-config file of the model suite",
#         default="",
#     )
#     parser.add_argument(
#         "--norm_elf",
#         default="/opt/vastai/vastpipe/data/elf/normalize",
#         help="image model elf file",
#     )
#     parser.add_argument(
#         "--space2depth_elf",
#         default="/opt/vastai/vastpipe/data/elf/space_to_depth",
#         help="image model elf file",
#     )
#     parser.add_argument(
#         "--txtmod_prefix",
#         default="/opt/vastai/vastpipe/data/models/clip_text-fp16-none-1_77-vacc/mod",
#         help="text model prefix of the model suite files",
#     )
#     parser.add_argument(
#         "--txtmod_hw_config",
#         help="text model hw-config file of the model suite",
#         default="",
#     )
#     parser.add_argument(
#         "--txtmod_vdsp_params",
#         default="./data/configs/clip_txt_vdsp.json",
#         help="text model vdsp preprocess parameter file",
#     )
#     parser.add_argument(
#         "-d",
#         "--device_id",
#         default=0,
#         type=int,
#         help="device id to run",
#     )
#     parser.add_argument(
#         "--input_file",
#         default="data/images/CLIP.png",
#         help="input file",
#     )
#     parser.add_argument(
#         "--label_file",
#         default="data/labels/imagenet.txt",
#         help="label file",
#     )
#     parser.add_argument(
#         "--dataset_filelist",
#         default="",
#         help="input dataset filelist",
#     )
#     parser.add_argument(
#         "--dataset_root",
#         default="",
#         help="input dataset root",
#     )
#     parser.add_argument(
#         "--dataset_output_file",
#         default="",
#         help="dataset output file",
#     )
#     parser.add_argument(
#         "--strings",
#         default="[a diagram,a dog,a cat]",
#         help='test strings, split by ","',
#     )
#     args = parser.parse_args()
#     return args


# if __name__ == "__main__":
#     args = argument_parser()
#     batch_size = 1
#     assert vsx.set_device(args.device_id) == 0
#     model = ClipModel(
#         args.imgmod_prefix,
#         args.norm_elf,
#         args.space2depth_elf,
#         args.txtmod_prefix,
#         args.txtmod_vdsp_params,
#         batch_size,
#         args.device_id,
#     )

#     if args.dataset_filelist == "":
#         image = cv2.imread(args.input_file)
#         assert image is not None, f"Failed to read input file: {args.input_file}"
#         texts = args.strings.strip("[").strip("]").split(",")
#         print(f"intput texts:{texts}")

#         result = model.process(image=image, texts=texts)
#         index = np.argsort(result)[::-1]
#         n = 5 if len(index) >=5 else len(index)
#         print(f"Top{n}:")
#         for i in range(n):
#             print(f"{i}th, string: {texts[index[i]]}, score: {result[index[i]]}")
#     else:
#         labels = load_labels(args.label_file)
#         texts_features = model.process_texts(labels)
#         filelist = []
#         with open(args.dataset_filelist, "rt") as f:
#             filelist = [line.strip() for line in f.readlines()]
#         with open(args.dataset_output_file, "wt") as fout:
#             for file in filelist:
#                 fullname = os.path.join(args.dataset_root, file)
#                 print(fullname)
#                 image = cv2.imread(fullname)
#                 assert image is not None, f"Failed to read input file: {fullname}"
#                 image_feature = model.process_image(image)
#                 result = model.post_process(image_feature, texts_features)
#                 index = np.argsort(result)[::-1]
#                 for i in range(5):
#                     fout.write(
#                         f"{file}: top-{i} id: {index[i]}, prob: {result[index[i]]}, class name: {labels[index[i]]}\n"
#                     )
