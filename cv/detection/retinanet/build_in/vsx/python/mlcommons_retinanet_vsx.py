# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import cv2
import time
import json
import glob
import torch
import argparse
import threading
import numpy as np
from PIL import Image
from tqdm.contrib import tzip
from queue import Queue
from threading import Thread, Event
from typing import Dict, Generator, Iterable, List, Union

import vaststreamx as vsx


class VSXInference:
    def __init__(self,
        model_prefix_path: Union[str, Dict[str, str]],
        vdsp_params_info: str,
        device_id: int = 0,
        batch_size: int = 1,
        is_async_infer: bool = False,
        model_output_op_name: str = "", ) -> None:


        if isinstance(vdsp_params_info, str):
            with open(vdsp_params_info) as f:
                vdsp_params_info_dict = json.load(f)
        
        self.model_size = vdsp_params_info_dict["OpConfig"]["OimageWidth"]
        self.device_id = device_id
        self.model_output_op_name = model_output_op_name
        self.preprocess_name = "preprocess_res"
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
        
        # 预处理算子输出
        n,c,h,w = self.model.input_shape[0]
        self.infer_stream.register_operator_output(self.preprocess_name, self.fusion_op, [[(c,h,w), vsx.TypeFlag.FLOAT16]])

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
                    pre_process_tensor = self.infer_stream.get_operator_output(self.preprocess_name)
                    # 输出顺序和输入一致 
                    self.current_id += 1
                    input_id,height, width = self.input_dict[self.current_id]
                    model_output_list = [
                        [vsx.as_numpy(out)[0].astype(np.float32) for out in result[0]]]
                    # self.post_processing(input_id, height, width, model_output_list)
                    self.result_dict[input_id] = model_output_list[0]
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break
    
    def finish(self,):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()

    def post_processing(self, input_id, height, width, stream_output_list):
        output_data = stream_output_list[0]
        self.result_dict[input_id] = output_data

    def _run(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        input_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)

        input_id = self.input_id
        self.input_dict[input_id] = (input_id, height, width)
        self.event_dict[input_id] = Event()


        self.infer_stream.run_async([input_image])
        
        self.input_id += 1

        return input_id

    def run(self, image:Union[str, np.ndarray]):
        input_id = self._run(image)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]

        return result

    def run_batch(self, images: Iterable[Union[str, np.ndarray]]) -> Generator[str, None, None]:

        queue = Queue(20)
        
        def input_thread():
            for image in images:
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
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        input_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)

        output = self.infer_stream.run_sync([input_image])
        model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in output[0]] ]
        output_data = model_output_list[0][0]

        return output_data


import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../..')
from source_code.model.boxes import box_iou, clip_boxes_to_image, batched_nms
from source_code.model.utils import Matcher, overwrite_eps, BoxCoder

def postprocess_detections(head_outputs, anchors, image_shapes, score_thresh=0.05, topk_candidates=1000, nms_thresh=0.5, detections_per_img=300):
    ## type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]    
    box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
    class_logits = head_outputs['cls_logits']
    box_regression = head_outputs['bbox_regression']

    num_images = len(image_shapes)

    detections = []

    for index in range(num_images):
        box_regression_per_image = [br[index] for br in box_regression]
        logits_per_image = [cl[index] for cl in class_logits]
        anchors_per_image, image_shape = anchors[index], image_shapes[index]

        image_boxes = []
        image_scores = []
        image_labels = []

        for box_regression_per_level, logits_per_level, anchors_per_level in \
                zip(box_regression_per_image, logits_per_image, anchors_per_image):
            num_classes = logits_per_level.shape[-1]

            # remove low scoring boxes
            scores_per_level = torch.sigmoid(logits_per_level).flatten()
            keep_idxs = scores_per_level > score_thresh
            scores_per_level = scores_per_level[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]

            # keep only topk scoring predictions
            num_topk = min(topk_candidates, topk_idxs.size(0))
            scores_per_level, idxs = scores_per_level.topk(num_topk)
            topk_idxs = topk_idxs[idxs]

            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode='floor')
            labels_per_level = topk_idxs % num_classes

            boxes_per_level = box_coder.decode_single(box_regression_per_level[anchor_idxs],
                                                            anchors_per_level[anchor_idxs])
            boxes_per_level = clip_boxes_to_image(boxes_per_level, image_shape)

            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        # non-maximum suppression
        keep = batched_nms(image_boxes, image_scores, image_labels, nms_thresh)
        keep = keep[:detections_per_img]

        detections.append({
            'boxes': image_boxes[keep],
            'scores': image_scores[keep],
            'labels': image_labels[keep],
        })

    return detections



def set_config():
    parse = argparse.ArgumentParser(description="RUN WITH VSX")
    parse.add_argument(
        "--image_dir",
        type=str,
        default="openimages/validation/data",
        help="image_dir img dir",
    )
    parse.add_argument(
        "--model_prefix_path",
        type=str,
        default="deploy_weights/resnext50_32x4d_fpn_forward_tvmlib_new-fp16-none-1_3_800_800-vacc/mod",
        help="model info")
    parse.add_argument(
        "--vdsp_params_info",
        type=str,
        default="vacc_deploy/vacc_code/vdsp_params/mlcommons-resnext50_32x4d_fpn-vdsp_params.json",
        help="vdsp op info",
    )
    parse.add_argument(
        "--onnx_file",
        type=str,
        default="resnext50_32x4d_fpn_head_outputs0_sim.onnx",
        help="onnx weight file for onnxruntime",
    )
    parse.add_argument(
        "--gt_file",
        type=str,
        default="openimages/annotations/openimages-mlperf.json",
        help="openimages-mlperf coco format json file",
    )
    parse.add_argument("--onnx_infer", action='store_true', default=False, help="whether to use onnxruntime inference")
    parse.add_argument("--draw_image", action='store_true', default=False, help="whether to draw detection boxes")
    parse.add_argument("--device_id", type=int, default=0, help="device id")
    parse.add_argument("--batch", type=int, default=1, help="bacth size")
    parse.add_argument("--save_dir", type=str, default="./vsx_results_vacc_fp16_libtvm_1231731", help="save result")
    args = parse.parse_args()

    return args


if __name__ == '__main__':
    args = set_config()

    # Test multiple images
    image_files = glob.glob(os.path.join(args.image_dir, "*.jpg"))
    # sort images
    image_files.sort()

    os.makedirs(args.save_dir, exist_ok=True)

    #############################################################################
    # define model
    if args.onnx_infer:
        print("onnxruntime infer...")
        import onnxruntime
        onnx_model = onnxruntime.InferenceSession(args.onnx_file)
        results = image_files
    else:
        print("vacc infer...")
        vsx_inference = VSXInference(model_prefix_path=args.model_prefix_path,
                            vdsp_params_info=args.vdsp_params_info,
                            device_id=args.device_id,
                            batch_size=args.batch,
                            is_async_infer = False,
                            model_output_op_name = "", 
                        )
        results = vsx_inference.run_batch(image_files)
    #############################################################################
    # some parameter for pre or post processing
    model_shape = [1, 3, 800, 800]
    n_classes = 264
    image_std = [0.229, 0.224, 0.225]
    image_mean = [0.485, 0.456, 0.406]
    from source_code.model.transform import GeneralizedRCNNTransform
    transform = GeneralizedRCNNTransform(image_size=[model_shape[-1], model_shape[-2]],
                                         image_mean=image_mean,
                                         image_std=image_std)
    transform.training = False

    from source_code.model.anchor_utils import AnchorGenerator
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    
    #################################################################################
    # load gt openimages-mlperf.json in coco format
    from pycocotools.coco import COCO
    openimages = COCO(args.gt_file)
    
    #################################################################################

    for (image_path, result) in tzip(image_files, results):
        # image_path = os.path.join(args.image_dir, '0009bad4d8539bb4.jpg')
        fin = open(os.path.join(args.save_dir, os.path.basename(image_path).replace('.jpg', '.txt')), 'w')
        #############################################################################
        # load image
        origin_image = cv2.imread(image_path)
        original_image_sizes = [(origin_image.shape[0], origin_image.shape[1])] # hw
        
        # pre-process
        image = origin_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = image / 255.0
        image = np.ascontiguousarray(image)
        image = np.expand_dims(image, axis=0).astype("float32")
        image = torch.from_numpy(image)
        trans_images, _ = transform(image, None)

        #############################################################################
        # forward
        # class branch
        all_cls_logits = []
        for i in range(0, 5):
            if args.onnx_infer:
                # for onnxruntime
                input_name = onnx_model.get_inputs()[0].name
                output_name = onnx_model.get_outputs()[i].name
                cls_logits = onnx_model.run([output_name], {input_name: np.array(trans_images.tensors)})[0]
            else:
                cls_logits = np.expand_dims(result[i], axis=0)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            cls_logits = torch.from_numpy(cls_logits)
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, n_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, n_classes)  # Size=(N, HWA, 4)
            all_cls_logits.append(cls_logits)
        
        all_cls_logits = torch.cat(all_cls_logits, dim=1)

        # box branch
        all_bbox_regression = []
        temp_bbox_regression = []
        for i in range(5, 10):
            if args.onnx_infer:
                # for onnxruntime
                input_name = onnx_model.get_inputs()[0].name
                output_name = onnx_model.get_outputs()[i].name
                bbox_regression = onnx_model.run([output_name], {input_name: np.array(trans_images.tensors)})[0]
            else:
                bbox_regression = np.expand_dims(result[i], axis=0)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            bbox_regression = torch.from_numpy(bbox_regression)
            temp_bbox_regression.append(bbox_regression)

            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)
            all_bbox_regression.append(bbox_regression)
        all_bbox_regression = torch.cat(all_bbox_regression, dim=1)

        head_outputs = {'cls_logits': all_cls_logits,
                        'bbox_regression': all_bbox_regression}
        #############################################################################

        #############################################################################
        # postprocess
        # create the set of anchors
        anchors = anchor_generator(trans_images, temp_bbox_regression)

        detections = []
        # recover level sizes
        num_anchors_per_level = [x.size(2) * x.size(3) for x in temp_bbox_regression]
        HW = 0
        for v in num_anchors_per_level:
            HW += v
        HWA = head_outputs['cls_logits'].size(1)
        A = HWA // HW
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

        # split outputs per level
        split_head_outputs = {}
        for k in head_outputs:
            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]
        
        # compute the detections
        detections = postprocess_detections(split_head_outputs, split_anchors, trans_images.image_sizes, score_thresh=0.05, nms_thresh=0.5)
        detections = transform.postprocess(detections, trans_images.image_sizes, original_image_sizes)

        # save text to eval and draw image
        for index, box in enumerate(detections[0]['boxes']):
            score = detections[0]['scores'][index].item()
            # if round(score, 5) < 0.25:
            #     continue
            label_id = detections[0]['labels'][index].item()
            label_name = openimages.cats[label_id]['name']
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            
            fin.write(f"{label_name} {score} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n")
            if args.draw_image:
                text = f"{label_name}: {round(score, 5)}"
                cv2.rectangle(origin_image, p1, p2, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(origin_image, text, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if args.draw_image:
            cv2.imwrite(os.path.join(args.save_dir, os.path.basename(image_path)), origin_image)
            # cv2.imwrite('0009bad4d8539bb4_vacc_int8.jpg', origin_image)
        fin.close()

    if not args.onnx_infer: 
        vsx_inference.finish()
    
"""

"""