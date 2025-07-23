# ==============================================================================
#
# Copyright (C) 2023 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author : andy.liu@vastaitech.com
@Time        :2023/08/10 15:33:02
'''

from loguru import logger
import cv2

import sys
import os
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../../..')
import vaststreamx as vsx
import numpy as np
import argparse
import glob
import os
import cv2 as cv
from typing import Dict, Generator, Iterable, List, Union
import json
from threading import Thread, Event
import time
from queue import Queue
from tqdm import tqdm

from source_code.tracker.byte_tracker import BYTETracker
from source_code.tracker.visualize import plot_tracking
from source_code.tracker.timer import Timer

import argparse
import os
import time

class Detector:
    def __init__(self,
                 model_prefix_path: Union[str, Dict[str, str]],
                 vdsp_params_info: str,
                 device_id: int = 0,
                 batch_size: int = 1,
                 balance_mode: int = 0,
                 is_async_infer: bool = False,
                 model_output_op_name: str = "", ) -> None:

        if isinstance(vdsp_params_info, str):
            with open(vdsp_params_info) as f:
                vdsp_params_info_dict = json.load(f)

        self.model_size = vdsp_params_info_dict["OpConfig"]["OimageWidth"]
        self.device_id = device_id
        self.model_output_op_name = model_output_op_name
        self.input_id = 0

        self.attr = vsx.AttrKey
        assert vsx.set_device(self.device_id)==0
        # 构建model，模型三件套目录
        model_path = model_prefix_path
        self.model = vsx.Model(model_path, batch_size)
        # 输入预处理op
        # RGB_PLANAR
        self.fusion_op = vsx.Operator.load_ops_from_json_file(vdsp_params_info)[0]

        # 构建graph
        self.graph = vsx.Graph()
        self.model_op = vsx.ModelOperator(self.model)
        self.graph.add_operators(self.fusion_op, self.model_op)

        # 构建stream
        self.infer_stream = vsx.Stream(self.graph, vsx.StreamBalanceMode.ONCE)
        if is_async_infer and len(self.model_output_op_name) > 0:
            self.infer_stream.register_operator_output(self.model_output_op_name, self.model_op)
        else:
            self.infer_stream.register_operator_output(self.model_op)

        self.infer_stream.build()

        self.current_id = -1
        self.input_dict = {}
        self.event_dict = {}
        self.result_dict = {}
        # 异步

    def run_sync(self, image: Union[str, np.ndarray]):
        ori_img = image.copy()
        image = np.stack(cv.split(cv.cvtColor(image, cv.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3

        nv12_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)
        yuv_nv12 = nv12_image  # vsx.cvtcolor(nv12_image, vsx.ImageFormat.YUV_NV12)
        # bgr_inter = vsx.create_image(cv_image, vsx.ImageFormat.BGR_INTERLEAVE, cv_image.shape[1], cv_image.shape[0], self.device_id)
        # yuv_nv12 = vsx.cvtcolor(bgr_inter, vsx.ImageFormat.RGB_PLANAR, vsx.ImageColorSpace.COLOR_SPACE_BT709)

        output = self.infer_stream.run_sync([yuv_nv12])
        model_output_list = [vsx.as_numpy(out)[0].astype(np.float32) for out in output[0]]
        box_ids = model_output_list[0]
        classes = len(box_ids)
        box_scores = model_output_list[1]
        # print('box_scores:\n', box_scores)

        # (classes, 4): xmin, ymin, xmax, ymax
        box_boxes = model_output_list[2]
        preds = np.concatenate((box_boxes, box_scores), axis=1)

        preds[:, :4] = scale_coords([608, 1088], preds[:, :4], ori_img.shape)

        return preds


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    # coords[:, :4] /= gain
    # clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("--path", default="./videos/palace.mp4", help="path to images or video")

    # detect args
    parser.add_argument("--model_prefix_path", type=str, default="deploy_weights/official_bytetrack_run_stream_int8/mod", help="model info")
    parser.add_argument("--vdsp_params_info",type=str, default="../vacc_code/vdsp_params/official-bytetrack_tiny_mot17-vdsp_params.json",  help="vdsp op info",)
    parser.add_argument("--device_id", type=int, default=0, help="device id")
    parser.add_argument("--batch", type=int, default=1, help="bacth size")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=int, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area',type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--result_dir",type=str, default="result/track_eval", help="track_eval path",)
    return parser


def imageflow_demo(predict, vis_folder, current_time, detect_size, args):
    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, args.path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (int(width), int(height)))
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()

    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            timer.tic()
            outputs = predict.run_sync(frame)
            h, w, c = frame.shape
            online_targets = tracker.update(outputs, [h, w], detect_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_classes = None
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                # vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area:  # and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # online_classes.append(t.det_class)
            timer.toc()
            results.append(
                (frame_id + 1, online_tlwhs, online_ids, online_scores))
            online_im = plot_tracking(frame,
                                      online_tlwhs,
                                      online_ids,
                                      classes=online_classes,
                                      frame_id=frame_id + 1,
                                      fps=1. / 1.)
            vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1


def main(args):
    vis_folder = os.path.join(args.result_dir)
    os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    # Load model
    detect_size = (608, 1088)
    predict = Detector(model_prefix_path=args.model_prefix_path,
                        vdsp_params_info=args.vdsp_params_info,
                        device_id=args.device_id,
                        batch_size=args.batch,
                        balance_mode = 0,
                        is_async_infer = False,
                        model_output_op_name = "", 
                    )

    current_time = time.localtime()
    imageflow_demo(predict, vis_folder, current_time, detect_size, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
