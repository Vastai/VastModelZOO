
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
import math
import torch
import argparse
import threading
import numpy as np
from PIL import Image
from tqdm.contrib import tzip
from torch.nn import functional as F
from queue import Queue
from threading import Thread, Event
from typing import Dict, Generator, Iterable, List, Union

import vaststreamx as vsx

import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../../../') # 指向源仓库路径
from source_code.metrics import metrics_class_dict, StreamSegMetrics
seg_metrics = StreamSegMetrics(2)

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
                    model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in result[0]] ]
                    self.result_dict[input_id] = []
                    self.post_processing(input_id, height, width, model_output_list)
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
        output_data = stream_output_list[0][0]
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
        

def draw_matting(image, mask):
    """
    image (np.uint8) shape (H,W,3)
    mask  (np.float32) range from 0 to 1, shape (H,W)
    """
    mask = 255*(1.0-mask)
    mask = np.expand_dims(mask, axis=2)
    mask = np.tile(mask, (1,1,3))
    mask = mask.astype(np.uint8)
    image_alpha = cv2.add(image, mask)
    return image_alpha


def miou(logits, targets, eps=1e-6):
	"""
	logits: (torch.float32)  shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
    # using to unet, deeplabv3+
	"""
	outputs = torch.argmax(logits, dim=1, keepdim=True).type(torch.int64)
	targets = torch.unsqueeze(targets, dim=1).type(torch.int64)
	# outputs = torch.zeros_like(logits).scatter_(dim=1, index=outputs, src=torch.tensor(1.0)).type(torch.int8)
	outputs = torch.zeros_like(logits).scatter_(dim=1, index=outputs, src=torch.ones_like(logits)).type(torch.int8)

	# targets = torch.zeros_like(logits).scatter_(dim=1, index=targets, src=torch.tensor(1.0)).type(torch.int8)
	targets = torch.zeros_like(logits).scatter_(dim=1, index=targets, src=torch.ones_like(logits)).type(torch.int8)

	inter = (outputs & targets).type(torch.float32).sum(dim=(2,3))
	union = (outputs | targets).type(torch.float32).sum(dim=(2,3))
	iou = inter / (union + eps)
	return iou.mean()


def custom_bisenet_miou(logits, targets):
	"""
	logits: (torch.float32) (main_out, feat_os16_sup, feat_os32_sup) of shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		return miou(logits[0], targets)
	else:
		return miou(logits, targets)


def custom_pspnet_miou(logits, targets):
	"""
	logits: (torch.float32) (main_out, aux_out) of shape (N, C, H, W), (N, C, H/8, W/8)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		return miou(logits[0], targets)
	else:
		return miou(logits, targets)


def custom_icnet_miou(logits, targets):
	"""
	logits: (torch.float32)
		[train_mode] (x_124_cls, x_12_cls, x_24_cls) of shape
						(N, C, H/4, W/4), (N, C, H/8, W/8), (N, C, H/16, W/16)

		[valid_mode] x_124_cls of shape (N, C, H, W)

	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		targets = torch.unsqueeze(targets, dim=1)
		targets = F.interpolate(targets, size=logits[0].shape[-2:], mode='bilinear', align_corners=True)[:,0,...]
		return miou(logits[0], targets)
	else:
		return miou(logits, targets)
     

def set_config():
    parse = argparse.ArgumentParser(description="RUN WITH VSX")
    parse.add_argument(
        "--file_path",
        type=str,
        default="/path/to/PPM-100/image",
        help="img dir",
    )
    parse.add_argument(
        "--gt_path",
        type=str,
        default="/path/to/PPM-100/matte",
        help="mask_dir",
    )
    parse.add_argument(
        "--model_prefix_path",
        type=str,
        default="deploy_weights/official_modnet_run_stream_int8/mod",
        help="model info")
    parse.add_argument(
        "--vdsp_params_info",
        type=str,
        default="../vacc_code/vdsp_params/official-modnet-vdsp_params.json",
        help="vdsp op info",
    )
    parse.add_argument("--device_id", type=int, default=0, help="device id")
    parse.add_argument("--batch", type=int, default=1, help="bacth size")
    parse.add_argument("--save_dir", type=str, default="./vsx_results", help="save result")
    args = parse.parse_args()

    return args


if __name__ == '__main__':
    args = set_config()

    vsx_inference = VSXInference(model_prefix_path=args.model_prefix_path,
                        vdsp_params_info=args.vdsp_params_info,
                        device_id=args.device_id,
                        batch_size=args.batch,
                        is_async_infer = False,
                        model_output_op_name = "", 
                    )

    # Test multiple images
    image_files = glob.glob(os.path.join(args.file_path, "*.jpg"))

    os.makedirs(args.save_dir, exist_ok=True)

    results = vsx_inference.run_batch(image_files)
    #############################################################################
    # import onnxruntime
    # session = onnxruntime.InferenceSession("deploy_weights/model_best.onnx")
    # import torch
    # device = torch.device('cpu')
    # model  = torch.jit.load("hrnet_w48-512_512.torchscript.pt", map_location=device)
    # model = model.to(device)
    # model.eval()
    #############################################################################
    metrics_ins = {}
    metrics_data = {}
    metrics = ['mse', 'mad', 'grad', 'conn']

    # add seg_metrics

    for key in metrics:
        key = key.lower()
        metrics_ins[key] = metrics_class_dict[key]()
        metrics_data[key] = None

    input_size = [1, 3, 480, 288]
    mean = [127.5, 127.5, 127.5]
    std = [127.5, 127.5, 127.5]
    ious = []
    # results = image_files
    for (image_path, result) in tzip(image_files, results):
        #############################################################################
        src_image = cv2.imread(image_path)
        image = cv2.resize(src_image, (input_size[-1], input_size[-2]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = np.expand_dims(image, axis=0)

        ## for onnxruntime
        # input_name = session.get_inputs()[0].name
        # output_name = session.get_outputs()[0].name
        # result = session.run([output_name], {input_name: image})
        
        ## for torch.jit
        # with torch.no_grad():
        #     heatmap = model(torch.from_numpy(image))
        # result = np.squeeze(heatmap.detach().numpy())
        #############################################################################
        
        # draw matting
        pred = (np.squeeze(result[0]) * 255).astype('uint8')

        # eval
        gt = cv2.imread(os.path.join(args.gt_path, os.path.basename(image_path)), 0)
        gt = cv2.resize(gt, (input_size[3], input_size[2]), interpolation=cv2.INTER_AREA)
        save_path = os.path.join(args.save_dir, os.path.basename(image_path))
        con = np.concatenate([gt, pred], axis=1)
        #cv2.imwrite(save_path, con)

        for key in metrics_ins.keys():
            metrics_data[key] = metrics_ins[key].update(pred, gt, trimap=None)
        
        # add seg_metrics 
        gt[gt < 128] = 0
        gt[gt >= 128] = 1
        pred[pred < 128] = 0
        pred[pred >= 128] = 1
        seg_metrics.update(gt, pred)


    for key in metrics_ins.keys():
        metrics_data[key] = metrics_ins[key].evaluate()
    print("matting_metrics: \n", metrics_data)

    # add seg_metrics
    val_score = seg_metrics.get_results()
    print("seg_metrics: \n", seg_metrics.to_str(val_score))
    vsx_inference.finish()
    
