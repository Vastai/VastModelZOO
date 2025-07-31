
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import vaststreamx as vsx
from model_base import ModelBase
import numpy as np
import argparse
import glob
import os
import cv2 as cv
from typing import Dict, List, Union
import json
import time
import ctypes
import copy
import random
from tqdm import tqdm

colors = []

# 创建一个集合，以确保元素不重复
unique_elements = set()

while len(colors) < 300:
    val1 = random.randint(60, 255)
    val2 = random.randint(70, 255)
    val3 = random.randint(100, 255)
    element = [val1, val2, val3]

    # 检查元素是否唯一，如果唯一，则添加到结果列表中
    if tuple(element) not in unique_elements:
        colors.append(element)
        unique_elements.add(tuple(element))

device_id = 0
vsx.set_device(device_id)

parse = argparse.ArgumentParser(description="RUN Det WITH VSX")
parse.add_argument(
    "--file_path",
    type=str,
    default= "det_coco_val/",
    help="img or dir  path",
)
parse.add_argument("--model_prefix_path", type=str, default="../model_convert/deploy_weights/yolov8s-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod", help="model info")
parse.add_argument(
    "--vdsp_params_info",
    type=str,
    default="vdsp_params.json", 
    help="vdsp op info",
)
parse.add_argument(
    "--vdsp_custom_op",
    type=str,
    default="yolov8_seg_post_proc", 
    help="vdsp custom op",
)
parse.add_argument(
    "--label_txt", type=str, default="./coco.txt", help="label txt"
)
parse.add_argument("--device_id", type=int, default=0, help="device id")
parse.add_argument("--batch", type=int, default=1, help="bacth size")
parse.add_argument("--save_dir", type=str, default="./output/", help="save_dir")
args = parse.parse_args()


class image_shape_layout_t(ctypes.Structure):
    _fields_ = [
        ("height", ctypes.c_int),
        ("width", ctypes.c_int),
        ("h_pitch", ctypes.c_int),
        ("w_pitch", ctypes.c_int),
    ]


class yolov8_gt_threshold_op_t(ctypes.Structure):
    _fields_=[
        ("img_shape",image_shape_layout_t),
        ("threshold",ctypes.c_float),
    ]


class yolov8_seg_op_t(ctypes.Structure):
    _fields_=[
        ("model_in_shape",image_shape_layout_t),
        ("model_out_shape",image_shape_layout_t),
        ("origin_image_shape",image_shape_layout_t),
        ("k", ctypes.c_uint32),
        ("retina_masks", ctypes.c_uint32),
        ("max_detect_num", ctypes.c_uint32),
    ]


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    x = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90,
    ]
    return x


def pred_to_json(file_name, classes, result):
    from pycocotools.mask import encode  # noqa
    jdict = []

    def single_encode(x):
        """Encode predicted masks as RLE and append results to jdict."""
        rle = encode(np.asarray(x[:, :, None], order='F', dtype='uint8'))[0]
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    # pred_masks = np.transpose(result[0]['segmentation'], (2, 0, 1))
    # with ThreadPool(NUM_THREADS) as pool:
    #     rles = pool.map(single_encode, pred_masks)
    # rles = single_encode(pred_masks)

    file_name = os.path.basename(file_name)
    file_name = os.path.splitext(file_name)[0]
    image_id = int(file_name) if file_name.isnumeric() else file_name

    coco_num = coco80_to_coco91_class()

    box = []
    label = []
    score = []
    seg = []
    for r in result:
        label.append(coco_num[classes.index(r['label'])])
        box.append(r['bbox'])
        score.append(r['score'])
        seg.append(r['segmentation'])
    if len(box):
        box = np.array(box)  # x1y1wh
        # box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner

        for i in range(len(box.tolist())):
            jdict.append({
                "image_id": image_id,
                "category_id": label[i],
                "bbox": [x for x in box[i].tolist()],
                "score": score[i],
                'segmentation': single_encode(seg[i])
                }
            )
    return jdict


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


def vdsp_post_process(
    custom_op,
    stream_ouput_data,
    model_size,
    width,
    height,
    retina_mask=0,
    device_id=0,
    cv_image=None,
    class_list=None,
    save_path=None
):
    box_ids = stream_ouput_data[0][0]
    classes = len(box_ids)
    box_scores = stream_ouput_data[1][0]
    box_boxes = stream_ouput_data[2][0]
    box_boxes = np.reshape(box_boxes, (classes, 4))
    mask_in = stream_ouput_data[3][0]
    pred_mask = stream_ouput_data[4][0]

    model_in_height = model_size
    model_in_width = model_size
    model_out_height = 160
    model_out_width = 160
    image_height = height
    image_width = width
    max_detect_num = 300
    mask_ch_num = 32

    op_conf = yolov8_seg_op_t()
    op_conf_size = ctypes.sizeof(yolov8_seg_op_t)

    op_conf.model_in_shape.height = model_in_height
    op_conf.model_in_shape.width = model_in_width
    op_conf.model_in_shape.h_pitch = model_in_height
    op_conf.model_in_shape.w_pitch = model_in_width

    op_conf.model_out_shape.height = model_out_height
    op_conf.model_out_shape.width = model_out_width
    op_conf.model_out_shape.h_pitch = model_out_height
    op_conf.model_out_shape.w_pitch = model_out_width

    op_conf.origin_image_shape.height = image_height
    op_conf.origin_image_shape.width = image_width
    op_conf.origin_image_shape.h_pitch = image_height
    op_conf.origin_image_shape.w_pitch = image_width

    op_conf.k = mask_ch_num
    op_conf.retina_masks = retina_mask
    op_conf.max_detect_num = 300
    vsx_class_id = vsx.from_numpy(box_ids, device_id=device_id)
    vsx_scores = vsx.from_numpy(box_scores, device_id=device_id)
    vsx_in_bboxes = vsx.from_numpy(box_boxes, device_id=device_id)
    vsx_mat_mul_a = vsx.from_numpy(mask_in, device_id=device_id)
    vsx_mat_mul_b = vsx.from_numpy(pred_mask, device_id=device_id)

    custom_op.set_config(
        config_bytes=ctypes.string_at(ctypes.byref(op_conf), op_conf_size),
        config_size=op_conf_size,
    )

    mask_out_h = model_in_height
    mask_out_w = model_in_width
    if op_conf.retina_masks == 1:
        mask_out_h = image_height
        mask_out_w = image_width
    output_list = custom_op.execute(
        tensors=[
            vsx_class_id,
            vsx_scores,
            vsx_in_bboxes,
            vsx_mat_mul_a,
            vsx_mat_mul_b,
        ],
        output_info=[
            ([max_detect_num, 1], vsx.TypeFlag.FLOAT16),
            ([max_detect_num, 1], vsx.TypeFlag.FLOAT16),
            ([max_detect_num, 4], vsx.TypeFlag.FLOAT16),
            ([1, max_detect_num, mask_out_h, mask_out_w], vsx.TypeFlag.UINT8),
            ([2], vsx.TypeFlag.UINT32),
            (
                [
                    (max_detect_num + 1)
                    * max(
                        model_out_width * model_out_height, image_height * image_width
                    )
                ],
                vsx.TypeFlag.UINT8,
            ),
        ],
    )
    output_np = [vsx.as_numpy(out) for out in output_list]

    box_ids = output_np[0]
    box_scores = output_np[1]
    box_out = output_np[2]
    pred_mask = output_np[3][0]
    print("pred mask shape: " , pred_mask.shape)
    det_num = output_np[4][0]
    print("det num: ", det_num)

    # cvMask = np.zeros((image_height, image_width, 3))

    result_dict = []
    for i in range(det_num):
        # if box_ids[i] == -1:
        #     break

        seg = pred_mask[i]

        # cvtemp = np.zeros((image_height, image_width, 3))
        # cvtemp[:, :, 0] = seg * colors[i][0]
        # cvtemp[:, :, 1] = seg * colors[i][1]
        # cvtemp[:, :, 2] = seg * colors[i][2]
        # cvMask = cvMask + cvtemp

        cur_box_out = box_out[i].tolist()
        cur_box_out[2] = cur_box_out[2] - cur_box_out[0]
        cur_box_out[3] = cur_box_out[3] - cur_box_out[1]

        result_dict.append(
            {
                "category_id": int(box_ids[i]),
                "label": class_list[int(box_ids[i])],
                "bbox": cur_box_out,
                "score": box_scores[i].item(),
                "segmentation": seg,
            }
        )
    # cv.imwrite(os.path.join(save_path, 'mask.jpg'), cvMask + cv_image)
    return result_dict


class ModelCV(ModelBase):
    def __init__(
        self, model_prefix, vdsp_config, batch_size=1, device_id=0, hw_config=""
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)

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
        result = self.process([input])
        result = result[0]
        return result

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
        np_out = [
            [copy.deepcopy(vsx.as_numpy(o)) for o in out] for out in outputs
        ]  # fix
        # print(np_out)
        return np_out


if __name__ == '__main__':
    custom_op = vsx.CustomOperator(
        op_name="yolov8_seg_op",
        elf_file_path=args.vdsp_custom_op,
        input_output_num=[1, 1],
    )
    detector = ModelCV(model_prefix=args.model_prefix_path,
                        vdsp_config=args.vdsp_params_info,
                        device_id=args.device_id,
                        batch_size=args.batch, 
                    )
    with open(args.label_txt) as f:
        classes = [cls.strip() for cls in f.readlines()]
    # Test multiple images
    images = glob.glob(os.path.join(args.file_path, "*"))
    time_begin = time.time()

    jdict = []
    for image in tqdm(images):
        print(image)
        cv_image = cv.imread(image, cv.IMREAD_COLOR)
        # save_path = os.path.join('result', image.split('\\')[-1].split('.')[0])
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # if cv_image.shape[0] % 2 != 0 or cv_image.shape[1] % 2 != 0:
        #     cv_image = cv.resize(
        #         cv_image, (cv_image.shape[1] // 2 * 2, cv_image.shape[0] // 2 * 2)
        #     )
        image_ = np.stack(cv.split(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)))
        c, height, width = image_.shape
        print("height: " , height , "width: " , width)
        vsx_image = vsx.create_image(
            image_, vsx.ImageFormat.RGB_PLANAR, width, height, args.device_id
        )
        outputs = detector.process(vsx_image)

        result = vdsp_post_process(
            custom_op,
            outputs,
            640,
            width,
            height,
            retina_mask=1,
            cv_image=cv.imread(image, cv.IMREAD_COLOR),
            class_list=classes,
            # save_path=save_path
        )
        r = pred_to_json(image, classes, result)
        jdict.extend(r)
        # save_result(image, result, args.save_dir)
    with open("predictions.json", 'w') as f:
        json.dump(jdict, f)  # flatten and save
    
    time_end = time.time()

    print(
        f"\n{len(images)} images in {time_end - time_begin} seconds, ({len(images) / (time_end - time_begin)} images/second)\n"
    )
print("test over")
