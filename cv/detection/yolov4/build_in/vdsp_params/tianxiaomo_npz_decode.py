
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :    zpwen
@Email  :    algorithm@vastaitech.com
@Time   :    2025/04/24 16:17:38
'''

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
        vdsp_params_path: str,
        classes: Union[str, List[str]],
        conf_thres: float = 0.25,
        nms_thres: float =0.45
    ) -> None:
        if isinstance(vdsp_params_path, str):
            with open(vdsp_params_path) as f:
                vdsp_params_dict = json.load(f)

        if isinstance(classes, str):
            with open(classes) as f:
                classes = [cls.strip() for cls in f.readlines()]

        self.model_size = [vdsp_params_dict["config"][3]["Value"], vdsp_params_dict["config"][3]["Value"]] # h,w
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


    def postprocess(self, box_ids_np, scores_np, bboxes_np, label_txt, img, origin_img_path, file_name, save_dir, save_img):
        with open(label_txt) as f:
            classes_list = [cls.strip() for cls in f.readlines()]
        origin_img = cv2.imread(origin_img_path)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        box_ids_np = np.expand_dims(box_ids_np, axis=1)
        # box_ids_np = box_ids_np[[not np.all(box_ids_np[i] == -1) for i in range(box_ids_np.shape[0])], :]
        scores_np = np.expand_dims(scores_np, axis=1)
        # scores_np = scores_np[[not np.all(scores_np[i] == -1) for i in range(scores_np.shape[0])], :]
        # bboxes_np = np.squeeze(bboxes_np, axis=0)
        # bboxes_np = bboxes_np[[not np.all(bboxes_np[i] == -1) for i in range(bboxes_np.shape[0])], :]
        
        # 反转尺寸
        bboxes_np = self.scale_coords(img.shape[2:], bboxes_np, origin_img.shape).round()
        res_length = len(bboxes_np)
        COLORS = np.random.uniform(0, 255, size=(len(classes_list), 3))
        fin = open(f"{save_dir}/{os.path.splitext(file_name)[0]}.txt", "w")
        # 画框
        if res_length:
            for index in range(res_length):
                label = classes_list[box_ids_np[index, :][0].astype(np.int8)]
                score = scores_np[index, :][0]
                bbox = bboxes_np[index, :].tolist()
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
                    COLORS[box_ids_np[index, :][0].astype(np.int8)],
                    2,
                )
                fin.write(f"{label} {score} {int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])}\n")
            if save_img:
                cv2.imwrite(f"{save_dir}/{file_name}", origin_img)
        fin.close()

    def npz_decode(self, output_npz_file:str):

        output_0 = np.load(output_npz_file, allow_pickle=True)["output_0"]
        output_1 = np.load(output_npz_file, allow_pickle=True)["output_1"]
        output_2 = np.load(output_npz_file, allow_pickle=True)["output_2"]
        stream_ouput = (output_0, output_1, output_2)

        # decode & nms
        import sys
        _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
        sys.path.append(_cur_file_path + os.sep + '../..')
        from source_code.tianxiaomo.tool import utils
        output = utils.post_processing(None, self.conf_thres, self.nms_thres, stream_ouput)

        # scale & draw
        img = np.zeros((1, 3, 416, 416))
        if len(output[0]) > 0:
            output = np.array(output)
            box_ids_np = output[0][:, 6]
            scores_np  = output[0][:, 5]
            bboxes_np  = output[0][:, :4]
            
            height, width = img.shape[2:]
            bboxes_np_new = []
            x1 = bboxes_np[:, 0] * width
            y1 = bboxes_np[:, 1] * height
            x2 = bboxes_np[:, 2] * width
            y2 = bboxes_np[:, 3] * height
            bboxes_np_new.append(np.stack((x1, y1, x2, y2), axis=1))

            # rescale and draw
            self.postprocess(
                box_ids_np = box_ids_np,
                scores_np = scores_np,
                bboxes_np = bboxes_np_new[0],
                label_txt = args.label_txt,
                img = img,
                origin_img_path = image_path,
                file_name = os.path.basename(image_path),
                save_dir = args.save_dir,
                save_img = args.draw_image,
                )
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
        # default= "./eval/npz_datalist_coco_eval.txt",
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
        default="detection/yolov4/vacc_code/vdsp_params/tianxiaomo-yolov4_tiny-vdsp_params.json",
        help="vdsp_params file path",
    )
    parse.add_argument(
        "--label_txt", type=str, default="./eval/coco.txt", help="label txt"
    )
    parse.add_argument(
        "--draw_image", type=bool, default=False, help="save the draw image"
    )
    parse.add_argument("--save_dir", type=str, default="output_npz_draw", help="save_dir")
    args = parse.parse_args()

    decoder = Decoder(
        vdsp_params_path=args.vdsp_params_path,
        classes=args.label_txt,
        conf_thres=0.001,
        nms_thres=0.65
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
        
        result = decoder.npz_decode(npz_file)

"""
vacc fp16
{'bbox_mAP': 0.433, 'bbox_mAP_50': 0.655, 'bbox_mAP_75': 0.471, 'bbox_mAP_s': 0.207, 'bbox_mAP_m': 0.494, 'bbox_mAP_l': 0.622, 'bbox_mAP_copypaste': '0.433 0.655 0.471 0.207 0.494 0.622'}
"""