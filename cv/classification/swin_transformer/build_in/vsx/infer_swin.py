import os
import numpy as np
import cv2
import argparse

import vaststreamx as vsx

import utils as utils
from typing import Union, List

attr = vsx.AttrKey


class ModelBase:
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        do_copy=True,
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


class MobileVit(ModelCV):
    def __init__(
        self, model_prefix, vdsp_config, batch_size=1, device_id=0, hw_config=""
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)
        self.model_input_height, self.model_input_width = self.model_.input_shape[0][
            -2:
        ]
        self.resize_height = int(256.0 / 224 * self.model_input_height)
        self.fusion_op_ = None
        for op in self.preproc_ops_:
            if op.op_type >= 100:
                self.fusion_op_ = op.cast_to_buildin_operator()
                break
        assert self.fusion_op_ is not None, "Can't find fusion op in vdsp op json file"

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height = input_shape[-2]
        width = input_shape[-1]
        dummy = np.zeros((height, width, 3), dtype=dtype)
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            device_dummy = vsx.create_image(
                dummy, vsx.ImageFormat.BGR_INTERLEAVE, width, height, self.device_id_
            )
            return [device_dummy] * batch_size

    def compute_size(self, img_w, img_h, size):
        if isinstance(size, int):
            size_h, size_w = size, size
        elif len(size) < 2:
            size_h, size_w = size[0], size[0]
        else:
            size_h, size_w = size[-2:]

        r = max(size_w / img_w, size_h / img_h)

        new_w = int(r * img_w)
        new_h = int(r * img_h)
        return (new_w, new_h)

    def process_impl(self, inputs):
        outputs = []
        for input in inputs:
            resize_width, resize_height = self.compute_size(
                input.width, input.height, self.resize_height
            )
            left = (resize_width - self.model_input_width) // 2
            top = (resize_height - self.model_input_height) // 2
            self.fusion_op_.set_attribute(
                {
                    attr.IIMAGE_WIDTH: input.width,
                    attr.IIMAGE_HEIGHT: input.height,
                    attr.IIMAGE_WIDTH_PITCH: input.width,
                    attr.IIMAGE_HEIGHT_PITCH: input.height,
                    attr.RESIZE_WIDTH: resize_width,
                    attr.RESIZE_HEIGHT: resize_height,
                    attr.CROP_X: left,
                    attr.CROP_Y: top,
                }
            )
            model_outs = self.stream_.run_sync([input])[0]
            outputs.append(vsx.as_numpy(model_outs[0]).astype(np.float32))
        return outputs


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/path/to/mobilevit_s-fp16-none-1_3_224_224-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--vdsp_params",
        default="/path/to/mobilevit_rgbplanar.json",
        help="model prefix of the model suite files",
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
        default="/path/to/cat.jpg",
        help="input file",
    )
    parser.add_argument(
        "--label_file",
        default="/path/to/imagenet.txt",
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
        default="",
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

    model = MobileVit(
        args.model_prefix,
        args.vdsp_params,
        batch_size=batch_size,
        device_id=args.device_id,
    )
    image_format = model.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read input file: {args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        result = model.process(vsx_image)
        index = np.argsort(result)[0][::-1]
        print("Top5:")
        for i in range(5):
            print(
                f"{i}th, score: {result[0, index[i]]:.4f}, class name: {labels[index[i]]}"
            )
    else:
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]
        with open(args.dataset_output_file, "wt") as fout:
            for file in filelist:
                fullname = os.path.join(args.dataset_root, file)
                print(fullname)
                cv_image = cv2.imread(fullname)
                assert cv_image is not None, f"Failed to read input file: {fullname}"
                vsx_image = utils.cv_bgr888_to_vsximage(
                    cv_image, image_format, args.device_id
                )
                result = model.process(vsx_image)
                index = np.argsort(result)[0][::-1]
                for i in range(5):
                    fout.write(
                        f"{file}: top-{i} id: {index[i]}, prob: {result[0, index[i]]}, class name: {labels[index[i]]}\n"
                    )
