# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import vaststreamx as vsx
import numpy as np
from numpy.linalg import norm
import argparse
import glob
import os
import cv2 as cv
from typing import Dict, Generator, Iterable, List, Union
import json
from threading import Thread, Event
import time
from queue import Queue
import torch
from skimage.morphology import skeletonize
from tqdm import tqdm

parse = argparse.ArgumentParser(description="RUN Det WITH VSX")
parse.add_argument(
    "--file_path",
    type=str,
    default= "/path/to/ctw1500/test/text_image/",
    help="img or dir  path",
)
parse.add_argument("--model_prefix_path", type=str, default="deploy_weights/mmocr_text_snake_fp16/mod", help="model info")
parse.add_argument(
    "--vdsp_params_info",
    type=str,
    default="../build_in/vdsp_params/mmocr-textsnake_resnet50_fpn-vdsp_params.json", 
    help="vdsp op info",
)
parse.add_argument(
    "--save_dir", type=str, default="output/", help=""
)
parse.add_argument("--device_id", type=int, default=0, help="device id")
parse.add_argument("--batch", type=int, default=1, help="bacth size")
args = parse.parse_args()

class TextSnakeDetector:
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
        # self.device = vsx.set_device(self.device_id)
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
        self.graph.add_operators(self.fusion_op, self.model_op)#

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
        self.consumer = Thread(target=self.async_receive_infer)
        self.consumer.start()

    def async_receive_infer(self, ):
        while 1:
            try:
                result = None
                if len(self.model_output_op_name) > 0:
                    result = self.infer_stream.get_operator_output(self.model_output_op_name)
                else:
                    result = self.infer_stream.get_operator_output(self.model_op)
                if result is not None:
                    # 输出顺序和输入一致 
                    self.current_id += 1
                    input_id,height, width = self.input_dict[self.current_id]
                    model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in result[0]] ]
                    self.result_dict[input_id] = []
                    self.post_processing(input_id, model_output_list)
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break
    
    def post_processing(self, input_id, stream_output_list):

        preds = stream_output_list[0][0].astype(np.float32)

        self.result_dict[input_id].append(np.expand_dims(preds, 0))

    def finish(self,):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()

    def _run(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv.imread(image, cv.IMREAD_COLOR)
            if cv_image.shape[0]%2 != 0 or cv_image.shape[1]%2 != 0:
                cv_image = cv.resize(cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            image = np.stack(cv.split(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        nv12_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)
        yuv_nv12 = nv12_image#vsx.cvtcolor(nv12_image, vsx.ImageFormat.YUV_NV12)
        
        input_id = self.input_id
        self.input_dict[input_id] = (input_id, height, width)
        self.event_dict[input_id] = Event()


        self.infer_stream.run_async([yuv_nv12])
        
        self.input_id += 1

        return input_id

    def run(self, image:Union[str, np.ndarray]):
        input_id = self._run(image)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]

        return result

    def run_batch(self, images: Iterable[Union[str, np.ndarray]]) -> Generator[str, None, None]:
        # for img in images:
        #     yield self.run(img)
        
        queue = Queue(20)

        def input_thread():
            for image in tqdm(images):
                input_id = self._run(image)
                queue.put(input_id)
            queue.put(None)

        thread = Thread(target=input_thread)
        thread.start()
        while True:
            input_id = queue.get()
            if input_id is None:
                break
            self.event_dict[input_id].wait()
            result = self.result_dict.pop(input_id)
            del self.event_dict[input_id]
            yield result
    
    def run_sync(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv.imread(image, cv.IMREAD_COLOR)
            if cv_image.shape[0]%2 != 0 or cv_image.shape[1]%2 != 0:
                cv_image = cv.resize(cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            image = np.stack(cv.split(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        nv12_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)
        yuv_nv12 = nv12_image#vsx.cvtcolor(nv12_image, vsx.ImageFormat.YUV_NV12)
        # bgr_inter = vsx.create_image(cv_image, vsx.ImageFormat.BGR_INTERLEAVE, cv_image.shape[1], cv_image.shape[0], self.device_id)
        # yuv_nv12 = vsx.cvtcolor(bgr_inter, vsx.ImageFormat.RGB_PLANAR, vsx.ImageColorSpace.COLOR_SPACE_BT709)
        
        output = self.infer_stream.run_sync([yuv_nv12])
        model_output_list = [vsx.as_numpy(out)[0].astype(np.float32) for out in output[0]] 

        return model_output_list


def fill_hole(input_mask) -> np.array:
    """Fill holes in matrix.

        Input:
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0]]
        Output:
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0]]

    Args:
        input_mask (ArrayLike): The input mask.

    Returns:
        np.array: The output mask that has been filled.
    """
    input_mask = np.array(input_mask)
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool_)

    return ~canvas | input_mask

def _centralize(points_yx: np.ndarray,
                normal_cos: torch.Tensor,
                normal_sin: torch.Tensor,
                radius: torch.Tensor,
                contour_mask: np.ndarray,
                step_ratio: float = 0.03) -> np.ndarray:
    """Centralize the points.

    Args:
        points_yx (np.array): The points in yx order.
        normal_cos (torch.Tensor): The normal cosine of the points.
        normal_sin (torch.Tensor): The normal sine of the points.
        radius (torch.Tensor): The radius of the points.
        contour_mask (np.array): The contour mask of the points.
        step_ratio (float): The step ratio of the centralization.
            Defaults to 0.03.

    Returns:
        np.ndarray: The centralized points.
    """

    h, w = contour_mask.shape
    top_yx = bot_yx = points_yx
    step_flags = np.ones((len(points_yx), 1), dtype=np.bool_)
    step = step_ratio * radius * np.hstack([normal_cos, normal_sin])
    while np.any(step_flags):
        next_yx = np.array(top_yx + step, dtype=np.int32)
        next_y, next_x = next_yx[:, 0], next_yx[:, 1]
        step_flags = (next_y >= 0) & (next_y < h) & (next_x > 0) & (
            next_x < w) & contour_mask[np.clip(next_y, 0, h - 1),
                                        np.clip(next_x, 0, w - 1)]
        top_yx = top_yx + step_flags.reshape((-1, 1)) * step
    step_flags = np.ones((len(points_yx), 1), dtype=np.bool_)
    while np.any(step_flags):
        next_yx = np.array(bot_yx - step, dtype=np.int32)
        next_y, next_x = next_yx[:, 0], next_yx[:, 1]
        step_flags = (next_y >= 0) & (next_y < h) & (next_x > 0) & (
            next_x < w) & contour_mask[np.clip(next_y, 0, h - 1),
                                        np.clip(next_x, 0, w - 1)]
        bot_yx = bot_yx - step_flags.reshape((-1, 1)) * step
    centers = np.array((top_yx + bot_yx) * 0.5, dtype=np.int32)
    return centers

def _merge_disks(disks: np.ndarray, disk_overlap_thr: float) -> np.ndarray:
    """Merging overlapped disks.

    Args:
        disks (np.ndarray): The predicted disks.
        disk_overlap_thr (float): The radius overlap threshold for merging
            disks.

    Returns:
        np.ndarray: The merged disks.
    """
    xy = disks[:, 0:2]
    radius = disks[:, 2]
    scores = disks[:, 3]
    order = scores.argsort()[::-1]

    merged_disks = []
    while order.size > 0:
        if order.size == 1:
            merged_disks.append(disks[order])
            break
        i = order[0]
        d = norm(xy[i] - xy[order[1:]], axis=1)
        ri = radius[i]
        r = radius[order[1:]]
        d_thr = (ri + r) * disk_overlap_thr

        merge_inds = np.where(d <= d_thr)[0] + 1
        if merge_inds.size > 0:
            merge_order = np.hstack([i, order[merge_inds]])
            merged_disks.append(np.mean(disks[merge_order], axis=0))
        else:
            merged_disks.append(disks[i])

        inds = np.where(d > d_thr)[0] + 1
        order = order[inds]
    merged_disks = np.vstack(merged_disks)

    return merged_disks

def postprocess(pred_results, min_text_region_confidence=0.6, min_center_region_confidence=0.2, min_center_area=30, disk_overlap_thr=0.03, radius_shrink_ratio=1.03):
    pred_results = torch.Tensor(pred_results[0])
    assert pred_results.dim() == 3

    pred_results[:2, :, :] = torch.sigmoid(pred_results[:2, :, :])
    pred_results = pred_results.detach().cpu().numpy()

    pred_text_score = pred_results[0]
    pred_text_mask = pred_text_score > min_text_region_confidence
    pred_center_score = pred_results[1] * pred_text_score
    pred_center_mask = \
        pred_center_score > min_center_region_confidence
    pred_sin = pred_results[2]
    pred_cos = pred_results[3]
    pred_radius = pred_results[4]
    mask_sz = pred_text_mask.shape

    scale = np.sqrt(1.0 / (pred_sin**2 + pred_cos**2 + 1e-8))
    pred_sin = pred_sin * scale
    pred_cos = pred_cos * scale

    pred_center_mask = fill_hole(pred_center_mask).astype(np.uint8)
    center_contours, _ = cv.findContours(pred_center_mask, cv.RETR_TREE,
                                          cv.CHAIN_APPROX_SIMPLE)
    
    result = []

    for contour in center_contours:
        if cv.contourArea(contour) < min_center_area:
            continue
        instance_center_mask = np.zeros(mask_sz, dtype=np.uint8)
        cv.drawContours(instance_center_mask, [contour], -1, 1, -1)
        skeleton = skeletonize(instance_center_mask)
        skeleton_yx = np.argwhere(skeleton > 0)
        y, x = skeleton_yx[:, 0], skeleton_yx[:, 1]
        cos = pred_cos[y, x].reshape((-1, 1))
        sin = pred_sin[y, x].reshape((-1, 1))
        radius = pred_radius[y, x].reshape((-1, 1))

        center_line_yx = _centralize(skeleton_yx, cos, -sin, radius,
                                          instance_center_mask)
        y, x = center_line_yx[:, 0], center_line_yx[:, 1]
        radius = (pred_radius[y, x] * radius_shrink_ratio).reshape(
            (-1, 1))
        score = pred_center_score[y, x].reshape((-1, 1))
        instance_disks = np.hstack(
            [np.fliplr(center_line_yx), radius, score])
        instance_disks = _merge_disks(instance_disks,
                                           disk_overlap_thr)

        instance_mask = np.zeros(mask_sz, dtype=np.uint8)
        for x, y, radius, score in instance_disks:
            if radius > 1:
                cv.circle(instance_mask, (int(x), int(y)), int(radius), 1,
                           -1)
        contours, _ = cv.findContours(instance_mask, cv.RETR_TREE,
                                       cv.CHAIN_APPROX_SIMPLE)

        score = np.sum(instance_mask * pred_text_score) / (
            np.sum(instance_mask) + 1e-8)
        if (len(contours) > 0 and cv.contourArea(contours[0]) > 0
                and contours[0].size > 8):
            polygon = contours[0].flatten().tolist()
        
        if score > 0.5:
            result.append(polygon)
    return result


def rescale_polygon(polygon,
                    scale_factor,
                    mode: str = 'mul') -> np.ndarray:
    """Rescale a polygon according to scale_factor.

    The behavior is different depending on the mode. When mode is 'mul', the
    coordinates will be multiplied by scale_factor, which is usually used in
    preprocessing transforms such as :func:`Resize`.
    The coordinates will be divided by scale_factor if mode is 'div'. It can be
    used in postprocessors to recover the polygon in the original
    image size.

    Args:
        polygon (ArrayLike): A polygon. In any form can be converted
            to an 1-D numpy array. E.g. list[float], np.ndarray,
            or torch.Tensor. Polygon is written in
            [x1, y1, x2, y2, ...].
        scale_factor (tuple(int, int)): (w_scale, h_scale).
        model (str): Rescale mode. Can be 'mul' or 'div'. Defaults to 'mul'.

    Returns:
        np.ndarray: Rescaled polygon.
    """
    assert len(polygon) % 2 == 0
    assert mode in ['mul', 'div']
    polygon = np.array(polygon, dtype=np.float32)
    poly_shape = polygon.shape
    reshape_polygon = polygon.reshape(-1, 2)
    scale_factor = np.array(scale_factor, dtype=float)
    if mode == 'div':
        scale_factor = 1 / scale_factor
    polygon = (reshape_polygon * scale_factor[None]).reshape(poly_shape)
    return polygon


if __name__ == '__main__':
    os.makedirs(args.save_dir, exist_ok=True)
    detector = TextSnakeDetector(model_prefix_path=args.model_prefix_path,
                        vdsp_params_info=args.vdsp_params_info,
                        device_id=args.device_id,
                        batch_size=args.batch,
                        balance_mode = 0,
                        is_async_infer = False,
                        model_output_op_name = "", 
                    )
    if os.path.isfile(args.file_path):
        result = detector.run_sync(args.file_path)#run(args.file_path)
        print(f"{args.file_path} => {result}")
        print(result[0].shape)
    else:
        # Test multiple images
        images = glob.glob(os.path.join(args.file_path, "*"))
        time_begin = time.time()
        results = detector.run_batch(images)

        for (image, result) in zip(images, results):
            # print(f"{image} => {result}")
            ori_image = cv.imread(image)
            h, w, _  = ori_image.shape
            scale = [w / result[0].shape[3], h / result[0].shape[2]]
            result = postprocess(result[0])
            # print(result)
            f = open(os.path.join(args.save_dir, image.split('/')[-1].split('.')[0] + '.txt'), 'w')
            for i in range(len(result)):
                box = result[i]
                box = rescale_polygon(box, scale)
                b = [str(bb) for bb in box]
                f.writelines(' '.join(b) + '\n')
            
            f.close()

        time_end = time.time()

        print(
            f"\n{len(images)} images in {time_end - time_begin} seconds, ({len(images) / (time_end - time_begin)} images/second)\n"
        )
    detector.finish()
    print("test over")

