
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import cv2
import glob
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, Generator, Iterable, List, Union


class Decoder:
    def __init__(
        self,
        classes: Union[str, List[str]],
        conf_thres: float = 0.25,
        nms_thres: float = 0.45,
        input_shape: List[int] = [1, 3, 416, 416]
    ) -> None:

        if isinstance(classes, str):
            with open(classes) as f:
                classes = [cls.strip() for cls in f.readlines()]

        self.model_size = input_shape[2:]
        self.classes = classes
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain

        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self, boxes, shape):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def postprocess(self, stream_ouput, image_file, save_dir, draw_image=False, **kwargs):
        origin_img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        file_name = os.path.basename(image_file)

        # post proecess
        # decode
        import sys
        _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
        sys.path.append(_cur_file_path + os.sep + '../..')
        from source_code.bubbliiiing.utils import utils_bbox

        # 416 anchors
        anchors = [12, 16,  19, 36,  40, 28,  36, 75,  76, 55,  72, 146,  142, 110,  192, 243,  459, 401]
        if len(stream_ouput) < 3:
            # yolov4_tiny
            anchors = [10,14,  23,27,  37,58,  81,82,  135,169,  344,319]

        anchors = np.array(anchors).reshape(-1, 2)
        bbox_util = utils_bbox.DecodeBoxNP(anchors=anchors, num_classes=len(self.classes), input_shape=self.model_size)
        if len(stream_ouput) < 3:
            # yolov4_tiny
            bbox_util = utils_bbox.DecodeBoxNP(anchors=anchors, num_classes=len(self.classes), input_shape=self.model_size, anchors_mask=[[3, 4, 5], [1, 2, 3]])

        outputs = bbox_util.decode_box(stream_ouput)
        
        # nms
        image_shape = np.array(np.shape(origin_img)[0:2])
        results = bbox_util.non_max_suppression(np.concatenate(outputs, 1),
                                                num_classes=len(self.classes),
                                                input_shape=self.model_size,
                                                image_shape=image_shape,
                                                letterbox_image=True,
                                                conf_thres=self.conf_thres,
                                                nms_thres=self.nms_thres)
        if results[0] is None:
            fin = open(f"{save_dir}/{os.path.splitext(file_name)[0]}.txt", "w")
            fin.close()
            return 
        
        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]
        res_length = len(top_boxes)
        
        # 画框
        if res_length:
            COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
            fin = open(f"{save_dir}/{os.path.splitext(file_name)[0]}.txt", "w")
            for index in range(res_length):
                label = self.classes[top_label[index].astype(np.int8)]
                score = top_conf[index]
                bbox = top_boxes[index].tolist()
                bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
                fin.write(f"{label} {score} {int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])}\n")
                if draw_image:
                    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(origin_img, p1, p2, (0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    text = f"{label}: {round(score * 100, 2)}%"
                    y = int(int(bbox[1])) - 15 if int(int(bbox[1])) - 15 > 15 else int(int(bbox[1])) + 15
                    cv2.putText(
                        origin_img,
                        text,
                        (int(bbox[0]), y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[top_label[index].astype(np.int8)],
                        2,
                    )

            if draw_image:
                cv2.imwrite(f"{save_dir}/{file_name}", origin_img)
        fin.close()

    def npz_decode(self, input_image_path:str, output_npz_file:str, save_dir:str, draw_image:bool=True):

        output_0 = np.load(output_npz_file, allow_pickle=True)["output_0"]
        output_1 = np.load(output_npz_file, allow_pickle=True)["output_1"]
        try:
            # yolov4
            output_2 = np.load(output_npz_file, allow_pickle=True)["output_2"]
            stream_ouput = (output_0, output_1, output_2)
        except Exception as e:
            # yolov4_tiny
            stream_ouput = (output_0, output_1)

        stream_ouput = [torch.Tensor(value) for value in stream_ouput]

        # decode & scale & nms & draw
        self.postprocess(stream_ouput, input_image_path, save_dir, draw_image)

        return stream_ouput


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="vamp out decoding")
    parse.add_argument(
        "--input_image_dir",
        type=str,
        default="./eval/coco_val2017",
        help="input source image folder",
    )
    parse.add_argument(
        "--vamp_datalist_path",
        type=str,
        default="./eval/npz_datalist_coco_eval.txt",
        # default="./eval/npz_datalist_coco_eval.txt",
        help="vamp datalist folder, set as None in vamc run case",
    )
    parse.add_argument(
        "--vamp_output_dir",
        type=str,
        default="./vamc/vamp/0.2.0/outputs/yolov4",
        help="vamp output folder",
    )
    parse.add_argument(
        "--vdsp_params_path",
        type=str,
        default="detection/yolov4/build_in/vdsp_params/bubbliiiing-yolov4-vdsp_params.json",
        help="vdsp_params file path",
    )
    parse.add_argument(
        "--label_txt", type=str, default="./eval/coco.txt", help="label txt"
    )
    parse.add_argument(
        "--draw_image", type=bool, default=False, help="save the draw image"
    )
    parse.add_argument("--save_dir", type=str, default="./save_dir_npz_draw", help="save_dir")
    args = parse.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    decoder = Decoder(
        classes=args.label_txt,
        conf_thres=0.001,
        nms_thres=0.65,
        input_shape=[1, 3, 416, 416]
    )

    npz_files = glob.glob(os.path.join(args.vamp_output_dir + "/*.npz"))
    npz_files.sort()

    if args.vamp_datalist_path is not None:
        # vamp
        with open(args.vamp_datalist_path, 'r') as f:
            input_npz_files = f.readlines()
    else:
        # vamc
        input_npz_files = [os.path.basename(file).replace(".npz", ".jpg") for file in npz_files]


    for index, npz_file in enumerate(tqdm(npz_files)):
        image_path = os.path.join(args.input_image_dir, os.path.basename(input_npz_files[index].strip().replace(".npz", ".jpg")))
        
        result = decoder.npz_decode(image_path, npz_file, save_dir=args.save_dir, draw_image=args.draw_image)

"""
yolov4 vacc fp16
{'bbox_mAP': 0.427, 'bbox_mAP_50': 0.653, 'bbox_mAP_75': 0.46, 'bbox_mAP_s': 0.2, 'bbox_mAP_m': 0.488, 'bbox_mAP_l': 0.618, 'bbox_mAP_copypaste': '0.427 0.653 0.460 0.200 0.488 0.618'}

yolov4_tiny vacc fp16
{'bbox_mAP': 0.205, 'bbox_mAP_50': 0.382, 'bbox_mAP_75': 0.2, 'bbox_mAP_s': 0.071, 'bbox_mAP_m': 0.241, 'bbox_mAP_l': 0.323, 'bbox_mAP_copypaste': '0.205 0.382 0.200 0.071 0.241 0.323'}
"""