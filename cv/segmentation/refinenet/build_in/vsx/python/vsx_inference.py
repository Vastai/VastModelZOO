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
import pandas as pd
from PIL import Image
from tqdm.contrib import tzip
from torch.nn import functional as F
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
        
def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def set_config():
    parse = argparse.ArgumentParser(description="RUN WITH VSX")
    parse.add_argument("--image_dir",type=str,default="/path/to/VOCdevkit/VOC2012/JPEGImages_val",help="img dir",)
    parse.add_argument("--mask_dir",type=str,default="/path/to/VOCdevkit/VOC2012/SegmentationClass",help="mask_dir",)
    parse.add_argument("--model_prefix_path",type=str,default="deploy_weights/official_refinenet_run_stream_fp16/mod",help="model info")
    parse.add_argument("--vdsp_params_info",type=str,default="../vacc_code/vdsp_params/drsleep-refinenet_lw_resnet50-vdsp_params.json",help="vdsp op info",)
    parse.add_argument("--color_txt", type = str, default = "../source_code/drsleep/voc2012_colors.txt", help = "colors")
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
    image_files = glob.glob(os.path.join(args.image_dir, "*.jpg"))
    os.makedirs(args.save_dir, exist_ok=True)

    import sys
    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + '../../..')
    from source_code.drsleep.score import SegmentationMetric
    
    input_size = [1, 3, 500, 500]
    metric = SegmentationMetric(21)

    results = vsx_inference.run_batch(image_files)

    for (image_path, result) in tzip(image_files, results):
        #############################################################################
        ori_image = Image.open(image_path)

        # post process
        tvm_predict = torch.from_numpy(result).unsqueeze(0)
        tvm_predict = F.interpolate(tvm_predict, (input_size[2], input_size[3]), mode='bilinear')     

        predict = tvm_predict[0].cpu().numpy()
        predict = np.asarray(predict, dtype="float32").transpose(1, 2, 0)

        predict_mask = cv2.resize(predict, ori_image.size, interpolation=cv2.INTER_CUBIC)

        colors = np.loadtxt(args.color_txt).astype('uint8')
        color = colorize(predict_mask.argmax(axis=2).astype(np.uint8), colors)
        color.save(os.path.join(args.save_dir, os.path.basename(image_path)+".png"))

        ########################################################################################################
        # eval
        label_path = os.path.join(args.mask_dir, os.path.basename(image_path).replace(".jpg", ".png"))
        if not os.path.exists(label_path):
            continue
        gt = Image.open(label_path)
        gt = gt.resize(size=(input_size[2], input_size[3]))
        target = np.array(gt).astype('int32')
        target[target == 255] = -1
        target = torch.from_numpy(target).long()

        metric.update(tvm_predict, target)
        pixAcc, mIoU = metric.get()
        print("{:s}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(image_path, pixAcc * 100, mIoU * 100))
        ########################################################################################################

    vsx_inference.finish()
    