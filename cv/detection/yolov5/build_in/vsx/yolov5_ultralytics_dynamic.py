# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import sys
import cv2
import ast
import argparse
import numpy as np
from typing import Union, List

import vaststreamx as vsx

attr = vsx.AttrKey

def load_labels(file):
    with open(file) as fin:
        return [line.strip() for line in fin.readlines()]

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

def cv_bgr888_to_vsximage(bgr888, vsx_format, device_id):
    h, w = bgr888.shape[:2]
    if vsx_format == vsx.ImageFormat.BGR_INTERLEAVE:
        res = bgr888
    elif vsx_format == vsx.ImageFormat.BGR_PLANAR:
        res = np.array(bgr888).transpose(2, 0, 1)
    elif vsx_format == vsx.ImageFormat.RGB_INTERLEAVE:
        res = cv2.cvtColor(bgr888, cv2.COLOR_BGR2RGB)
    elif vsx_format == vsx.ImageFormat.RGB_PLANAR:
        res = np.array(cv2.cvtColor(bgr888, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    elif vsx_format == vsx.ImageFormat.YUV_NV12:
        res = cv_bgr888_to_nv12(bgr888=bgr888)
    else:
        assert False, f"Unsupport format:{vsx_format}"
    return vsx.create_image(
        res,
        vsx_format,
        w,
        h,
        device_id,
    )

class DynamicModelBase:
    def __init__(
        self, module_info, vdsp_config, max_input_size, batch_size=1, device_id=0
    ) -> None:
        self.device_id_ = device_id
        assert (
            vsx.set_device(device_id) == 0
        ), f"set device failed, device_id={device_id}"
        self.model_ = vsx.Model(module_info)
        self.model_.set_input_shape(max_input_size)
        self.model_.set_batch_size(batch_size)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.preproc_ops_ = vsx.Operator.load_ops_from_json_file(vdsp_config)
        assert len(self.preproc_ops_) > 0, "load_ops_from_json_file failed"
        self.graph_ = vsx.Graph()
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
                    return imagetype_to_vsxformat(imagetype)
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
        else:
            if isinstance(input, np.ndarray):
                return self.process(cv_rgb_image_to_vastai(input, self.device_id_))
            else:
                return self.process_impl([input])[0]

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height, width = input_shape[-2:]
        dummy = np.zeros((height, width, 3), dtype=dtype)
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            vacc_dummy = vsx.create_image(
                dummy, vsx.ImageFormat.BGR_INTERLEAVE, width, height, self.device_id_
            )
            return [vacc_dummy] * batch_size

    def set_model_input_shape(self, model_input_shape):
        self.model_input_shape_ = model_input_shape

    def process_impl(self, inputs):
        assert len(inputs) == len(self.model_input_shape_)
        input_shapes = [[shape] for shape in self.model_input_shape_]
        outputs = self.stream_.run_sync(
            inputs, {"dynamic_model_input_shapes": input_shapes}
        )
        return [
            self.post_process(
                output, inputs[i].width, inputs[i].height, self.model_input_shape_[i]
            )
            for i, output in enumerate(outputs)
        ]

    def set_threshold(self, threshold):
        self.threshold_ = threshold

    def post_process(self, out_tensors, image_width, image_height, model_input_shape):
        data_count = out_tensors[0].size
        result_np = np.zeros((data_count, 6), dtype=np.float32) - 1
        # check tensor size validation
        assert (
            out_tensors[0].size == out_tensors[1].size
            and out_tensors[1].size * 4 == out_tensors[2].size
        ), f"Output tensor size error, sizes are:{out_tensors[0].size},{out_tensors[1].size},{out_tensors[2].size}"
        class_data = vsx.as_numpy(out_tensors[0]).squeeze()
        score_data = vsx.as_numpy(out_tensors[1]).squeeze()
        bbox_data = vsx.as_numpy(out_tensors[2]).squeeze()

        model_height, model_width = model_input_shape[-2:]

        r = min(model_width / image_width, model_height / image_height)
        unpad_w = image_width * r
        unpad_h = image_height * r
        dw = (model_width - unpad_w) / 2
        dh = (model_height - unpad_h) / 2

        for i in range(data_count):
            category = int(class_data[i])
            if category < 0:
                break
            score = score_data[i]
            if score > self.threshold_:
                bbox_xmin = (bbox_data[i][0] - dw) / r
                bbox_ymin = (bbox_data[i][1] - dh) / r
                bbox_xmax = (bbox_data[i][2] - dw) / r
                bbox_ymax = (bbox_data[i][3] - dh) / r
                bbox_width = bbox_xmax - bbox_xmin
                bbox_height = bbox_ymax - bbox_ymin
                result_np[i][0] = category
                result_np[i][1] = score
                result_np[i][2] = bbox_xmin
                result_np[i][3] = bbox_ymin
                result_np[i][4] = bbox_width
                result_np[i][5] = bbox_height
        return result_np



def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--module_info",
        default="/path/to/yolov5s-dynamic_module_info.json",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--vdsp_params",
        default="/path/to/yolo_div255_bgr888.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--threshold",
        default=0.01,
        type=float,
        help="device id to run",
    )
    parser.add_argument(
        "--max_input_shape",
        default="[1,3,640,640]",
        help="model max input shape",
    )
    parser.add_argument(
        "--label_file",
        default="/path/to/coco.txt",
        help="label file",
    )
    parser.add_argument(
        "--input_file",
        default="/path/to/det_coco_val/000000001503.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="dynamic_result.jpg",
        help="output file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="/path/to/det_coco_val", 
        help="dataset filename list",
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="dataset root",
    )
    parser.add_argument(
        "--dataset_output_folder",
        default="vsx_output-resize",
        help="dataset output folder path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    os.makedirs(args.dataset_output_folder, exist_ok=True)

    labels = load_labels(args.label_file)
    max_input_shape = ast.literal_eval(args.max_input_shape)
    batch_size = 1
    dynamic_model = DynamicModelBase(
        args.module_info,
        args.vdsp_params,
        [max_input_shape],
        batch_size,
        args.device_id,
    )
    model_min_input_size, model_max_input_size= 416,640
    dynamic_model.set_threshold(args.threshold)
    image_format = dynamic_model.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        image = cv2.imread(args.input_file)
        assert image is not None, f"Failed to open {args.input_file}"
        ori_h, ori_w, c = image.shape
        input_size = ori_w if ori_w > ori_h else ori_h
        if input_size %2 !=0:
            input_size += 1
        if input_size > model_max_input_size:
            input_size = model_max_input_size
        elif input_size < model_min_input_size:
            input_size = model_min_input_size
        dynamic_model.set_model_input_shape([[1, 3, input_size, input_size]])
        vsx_image = cv_bgr888_to_vsximage(image, image_format, args.device_id)
        objects = dynamic_model.process(vsx_image)
        print("Detection objects:")
        for obj in objects:
            if obj[0] > 0:
                bbox = [int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])]
                print(
                    f"Object class: {labels[int(obj[0])]}, score: {obj[1]}, bbox: {bbox}"
                )
            else:
                break
        if args.output_file != "":
            for obj in objects:
                if obj[0] > 0:
                    cv2.rectangle(
                        image,
                        pt1=(int(obj[2]), int(obj[3])),
                        pt2=(int(obj[2] + obj[4]), int(obj[3] + obj[5])),
                        color=(0, 0, 255),
                        thickness=1,
                    )
                else:
                    break
            cv2.imwrite(args.output_file, image)
    else:
        filelist = []
        # with open(args.dataset_filelist, "rt") as f:
        #     filelist = [line.strip() for line in f.readlines()]
        import glob
        from tqdm import tqdm

        filelist = glob.glob(os.path.join(args.dataset_filelist, "*.jpg"))
        

        for image_file in tqdm(filelist):
            fullname = os.path.join(args.dataset_root, image_file)
            image = cv2.imread(fullname)
            assert image is not None, f"Failed to open {fullname}"
            ori_h, ori_w, c = image.shape
            input_size = ori_w if ori_w > ori_h else ori_h
            if input_size % 2 !=0:
                input_size += 1
            if input_size > model_max_input_size:
                input_size = model_max_input_size
            elif input_size < model_min_input_size:
                input_size = model_min_input_size
            dynamic_model.set_model_input_shape([[1, 3, input_size, input_size]])
            vsx_image = cv_bgr888_to_vsximage(image, image_format, args.device_id)
            objects = dynamic_model.process(vsx_image)
            base_name, _ = os.path.splitext(os.path.basename(image_file))
            outfile = open(
                os.path.join(args.dataset_output_folder, base_name + ".txt"), "wt"
            )
            print(f"{image_file} detection objects:")
            COLORS = np.random.uniform(0, 255, size=(80, 3))

            for obj in objects:
                if obj[1] >= 0:
                    bbox = [int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])]
                    print(
                        f"Object class: {labels[int(obj[0])]}, score: {obj[1]}, bbox: {bbox}"
                    )
                    outfile.write(
                        f"{labels[int(obj[0])]} {obj[1]} {obj[2]:.3f} {obj[3]:.3f} {(obj[2]+obj[4]):.3f} {(obj[3]+obj[5]):.3f}\n"
                    )
                    text = f"{labels[int(obj[0])]}: {obj[1]}"
                    # cv2.rectangle(image, (int(obj[2]), int(obj[3])), (int(obj[2]+obj[4]), int(obj[3]+obj[5])), (0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

                    # cv2.putText(
                    #     image,
                    #     text,
                    #     (max(0, int(obj[2])), int(obj[3])),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.5,
                    #     COLORS[int(obj[0])],
                    #     2,
                    # )
                    # cv2.imwrite(os.path.join(args.dataset_output_folder, base_name + ".jpg"), image)
                else:
                    break
            outfile.close()
