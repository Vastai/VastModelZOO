import os
import numpy as np
import cv2
import argparse

import ctypes
from enum import Enum
import numpy as np
from typing import Union, List
import vaststreamx as vsx
from tqdm import tqdm


def convert_uint16_to_float(a):
    sign = a>>15
    exp = (a>>10) & 31
    frac = a & 1023
    if exp == 0:
        return (-1)**sign*2**-14*(0+frac/1024.0)
    elif exp == 31 and frac==0:
        return float('inf')*(-1)**sign
    elif exp == 31:
        return float('nan')
    else:
        return (-1)**sign*2**(exp-15)*(1+frac/1024.0)

def convert_float_to_uint16(a):
    sign = int(a<0)
    exp = int(np.log2(int(abs(a))))
    frac = int((abs(a) / 2**exp - 1)*1024)
    return sign*2**15 + (exp+15)*1024 + frac

def cv_rgb_image_to_vastai(image_cv, device_id):
    assert len(image_cv.shape) >= 2
    h = image_cv.shape[0]
    w = image_cv.shape[1]
    if len(image_cv.shape) == 3:
        return vsx.create_image(
            image_cv, vsx.ImageFormat.BGR_INTERLEAVE, w, h, device_id
        )
    elif len(image_cv.shape) == 2:
        return vsx.create_image(image_cv, vsx.ImageFormat.GRAY, w, h, device_id)
    else:
        raise Exception("unsupported ndarray shape", image_cv.shape)

class NormalType(Enum):
    NORMAL_EQUAL=0                    #< R = R*scale_R; R/G/B is the same process as R;
    NORMAL_MINUSMEAN_DIVSTD=1         #< R = R*scale_R/std -mean_R*scale_R/std; R/G/B is the same process as R;
    NORMAL_DIV255_MINUSMEAN_DIVSTD=2  #< R = R*scale_R/(255*std) -mean_R*scale_R/std; R/G/B is the same process as R;
    NORMAL_DIV127_5_MINUSONE=3        #< R = R*scale_R/127.5 -1*scale_R; R/G/B is the same process as R;
    NORMAL_DIV255=4                   #< R = R*scale_R/255; R/G/B is the same process as R;

MAX_NORMA_CH_NUM = 4
class normalize_cfg_t(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),                         #< width of the image
        ("height", ctypes.c_uint32),                        #< height of the image
        ("in_width_pitch", ctypes.c_uint32),                #< input width pitch of the image, in pixel
        ("in_height_pitch", ctypes.c_uint32),               #< input height pitch of the image, in pixel
        ("out_width_pitch", ctypes.c_uint32),               #< input height pitch of the image, in pixel
        ("out_height_pitch", ctypes.c_uint32),              #< input height pitch of the image, in pixel
        ("ch_num", ctypes.c_uint32),                        #< channel number; should be 1, 2, 3, or 4
        ("norma_type", ctypes.c_int32),                     #< normalization type; see resize_normal_quant_api.h for details
        ("mean", ctypes.c_uint16 * MAX_NORMA_CH_NUM),       #< mean of each channel; datatype is fp16
        ("std", ctypes.c_uint16 * MAX_NORMA_CH_NUM),        #< std of each channel; datatype is fp16
    ]

class DataTypeEnum(Enum):
    DATA_TYPE_FP16 = 0
    DATA_TYPE_INT8 = 1
    DATA_TYPE_BFP16 = 2
    DATA_TYPE_FLOAT32 = 3
    DATA_TYPE_DOUBLE = 4
    DATA_TYPE_INT32 = 5
    DATA_TYPE_RESERVED = 6

class LayoutTypeEnum(Enum):
    NCHW_layout = 0
    MatrixA_layout = 1

class space_to_depth_cfg_t(ctypes.Structure):
    _fields_ = [
        ("input_dims_num", ctypes.c_uint32),
        ("input_dims", ctypes.c_uint32 * 4),
        ("input_align_dims", ctypes.c_uint32 * 4),
        ("input_layout", ctypes.c_int32),           # NCHW, N = 1
        ("input_type", ctypes.c_int32 ),
        ("kh", ctypes.c_uint32),                    #block_size height.
        ("kw", ctypes.c_uint32),                    #block size width.
        ("output_dims_num", ctypes.c_uint32),
        ("output_dims", ctypes.c_uint32 * 4),
        ("output_align_dims", ctypes.c_uint32 * 4),
        ("output_layout", ctypes.c_int32),          # MatrixA
    ]


class PreProcessOp:
    def __init__(
        self,
        norm_elf_file,
        space_to_depth_elf_file,
        model_input_shape = [1,3,224,224],
        block_shape = [16,16],
        batch_size=1,
        device_id=0
    ) -> None:
        self.device_id_ = device_id

        self.input_n_num = 1
        self.input_c_num = model_input_shape[-3]
        self.resize_height, self.resize_width = model_input_shape[-2], model_input_shape[-1]
        self.resize_shape = [self.resize_height, self.resize_width]

        self.norm_height, self.norm_width = [self.resize_height, self.resize_width]
        self.norm_op_ = vsx.CustomOperator("opf_normalize", norm_elf_file)
        self.set_norm_cfg_()

        self.block_height, self.block_width = block_shape
        self.output_height, self.output_width, self.output_height_pitch, self.output_width_pitch = \
            self.calc_space_to_depth_out_shape_()
        self.space_to_depth_op_ = vsx.CustomOperator("opf_space_to_depth_out_matrix", space_to_depth_elf_file)
        self.set_space_to_depth_cfg_()

    def set_norm_cfg_(self):
        op_param = normalize_cfg_t()
        op_param.width = self.resize_width
        op_param.height = self.resize_height
        op_param.in_width_pitch = self.resize_width
        op_param.in_height_pitch = self.resize_height
        op_param.out_width_pitch = self.norm_width
        op_param.out_height_pitch = self.norm_height
        op_param.ch_num = self.input_c_num
        op_param.norma_type = NormalType.NORMAL_DIV127_5_MINUSONE.value
        op_param.mean[0] = 22520  # NormalType.NORMAL_DIV127_5_MINUSONE don't use mean std
        op_param.mean[1] = 22520
        op_param.mean[2] = 22520
        op_param.std[0] = 22520
        op_param.std[1] = 22520
        op_param.std[2] = 22520
        op_conf_size = ctypes.sizeof(normalize_cfg_t)
        self.norm_op_.set_config(
            config_bytes=ctypes.string_at(ctypes.byref(op_param), op_conf_size),
            config_size=op_conf_size,
        )

    def is_power_of_two_(self, num):
        return (num & (num - 1) == 0) and (num & -num == num)

    def calc_space_to_depth_out_shape_(self):
        c = self.input_c_num
        h = self.norm_height
        w = self.norm_width
        width = int(c*self.block_height * self.block_width)
        height = int(c*h*w/width)
        width_pitch = width if self.is_power_of_two_(width) else 2**(int(np.log2(width))+1)
        height_pitch = height if height%16==0 else 16*(height//16+1)
        return height, width, height_pitch, width_pitch

    def set_space_to_depth_cfg_(self):
        op_param = space_to_depth_cfg_t()
        op_param.input_dims_num = 4
        op_param.input_dims[0] = self.input_n_num                   #N 1
        op_param.input_dims[1] = self.input_c_num                   #C 3
        op_param.input_dims[2] = self.norm_height                   #H 224
        op_param.input_dims[3] = self.norm_width                    #W 224
        op_param.input_align_dims[0] = self.input_n_num
        op_param.input_align_dims[1] = 4
        op_param.input_align_dims[2] = self.norm_height
        op_param.input_align_dims[3] = self.norm_width
        op_param.input_layout = LayoutTypeEnum.NCHW_layout.value
        op_param.input_type = DataTypeEnum.DATA_TYPE_FP16.value
        op_param.kh = self.block_height                             #16
        op_param.kw = self.block_width                              #16
        op_param.output_dims_num = 2
        op_param.output_dims[0] = self.output_height                #196
        op_param.output_dims[1] = self.output_width                 #768
        op_param.output_align_dims[0] = self.output_height_pitch    #208
        op_param.output_align_dims[1] = self.output_width_pitch     #1024
        op_param.output_layout = LayoutTypeEnum.MatrixA_layout.value
        op_conf_size = ctypes.sizeof(space_to_depth_cfg_t)
        self.space_to_depth_op_.set_config(
            config_bytes=ctypes.string_at(ctypes.byref(op_param), op_conf_size),
            config_size=op_conf_size,
        )

    def process(
        self, input: Union[List[np.ndarray], List[vsx.Tensor], np.ndarray, vsx.Tensor]
    ):
        if isinstance(input, list):
            if isinstance(input[0], np.ndarray):
                return self.process(
                    [
                        cv_rgb_image_to_vastai(x, self.device_id_)
                        for x in input
                    ]
                )
            else:
                return self.process_impl(input)
        return self.process([input])[0]

    def process_impl(self, inputs):
        outputs = []
        for input in inputs:
            img_rgbplanar = vsx.cvtcolor(input, vsx.ImageFormat.RGB_PLANAR)
            resize_out = vsx.resize(img_rgbplanar, vsx.ImageResizeType.BILINEAR_PILLOW, self.resize_width, self.resize_height)
            norm_out = self.norm_op_.execute(
                tensors=[resize_out],
                output_info=[([self.input_c_num, self.norm_height, self.norm_width], vsx.TypeFlag.FLOAT16)],
            )
            outs = self.space_to_depth_op_.execute(
                tensors=norm_out,
                output_info=[([self.output_height_pitch, self.output_width_pitch], vsx.TypeFlag.FLOAT16)],
            )
            outputs.append(outs)
        return outputs


class VIT:
    def __init__(
        self,
        model_prefix,
        norm_elf_file = "",
        space_to_depth_elf_file = "",
        block_shape=[16,16],
        batch_size=1,
        device_id=0,
        hw_config=""
    ) -> None:
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.graph_ = vsx.Graph(output_type=vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_884_DEVICE)
        self.graph_.add_operators(self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

        model_input_shape = self.model_.input_shape[0]
        self.ops = PreProcessOp(
            norm_elf_file, space_to_depth_elf_file, model_input_shape, block_shape,
            device_id=self.device_id_
        )

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
        img_preprocess = self.ops.process(input)
        outputs=self.stream_.run_sync(img_preprocess)
        # take first 1000 valid value
        t = outputs[0][0]
        return [[vsx.as_numpy(o)[0:1000] for o in out] for out in outputs]


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vastpipe/data/models/vit-b-fp16-none-1_3_224_224-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--norm_elf_file",
        default="/opt/vastai/vastpipe/data/elf/normalize",
        help="normalize op elf file",
    )
    parser.add_argument(
        "--space_to_depth_elf_file",
        default="/opt/vastai/vastpipe/data/elf/space_to_depth",
        help="space_to_depth op elf files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--input_file",
        default="../../../data/images/cat.jpg",
        help="input file",
    )
    parser.add_argument(
        "--label_file",
        default="../../../data/labels/imagenet.txt",
        help="label file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="dataset filelst",
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="input dataset root",
    )
    parser.add_argument(
        "--dataset_output_file",
        default="runstream_result.txt",
        help="dataset output file",
    )
    args = parser.parse_args()
    return args

def load_labels(file):
    with open(file) as fin:
        return [line.strip() for line in fin.readlines()]

if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    labels = load_labels(args.label_file)

    classifier = VIT(
        args.model_prefix,
        args.norm_elf_file,
        args.space_to_depth_elf_file,
        block_shape=[16,16],
        batch_size=batch_size,
        device_id=args.device_id
    )

    if args.dataset_filelist == "":
        image = cv2.imread(args.input_file)
        assert image is not None, f"Failed to read input file: {args.input_file}"
        output = classifier.process(image)
        result = np.array(output)
        index = np.argsort(result)[0][::-1]
        print("Top5:")
        for i in range(5):
            print(
                f"top-{i} id: {index[i]}, prob: {result[0, index[i]]}, class name: {labels[index[i]]}"
            )
    else:
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]
        with open(args.dataset_output_file, "wt") as fout:
            for file in tqdm(filelist):
                fullname = os.path.join(args.dataset_root, file)
                image = cv2.imread(fullname)
                assert image is not None, f"Failed to read input file: {fullname}"
                output = classifier.process(image)
                result = np.array(output)
                index = np.argsort(result)[0][::-1]
                for i in range(5):
                    fout.write(
                        f"{file}: top-{i} id: {index[i]}, prob: {result[0, index[i]]}, class name: {labels[index[i]]}\n"
                    )

# eval 1k
# [VACC]:  top1_rate: 81.8 top5_rate: 95.9
