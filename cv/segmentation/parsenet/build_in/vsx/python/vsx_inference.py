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
from tqdm import tqdm

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
            if cv_image.shape[0]%2 != 0 or cv_image.shape[1]%2 != 0:
                cv_image = cv2.resize(cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
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
        

def tenor2mask(tensor_data):
    MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], 
                     [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    
    if len(tensor_data.shape) < 4:
        tensor_data = tensor_data.unsqueeze(0)
    if tensor_data.shape[1] > 1:
        tensor_data = tensor_data.argmax(dim=1) 

    tensor_data = tensor_data.squeeze(1).data.cpu().numpy()
    color_maps = []
    for t in tensor_data:
        tmp_img = np.zeros(tensor_data.shape[1:] + (3,))
        # tmp_img = np.zeros(tensor_data.shape[1:])
        for idx, color in enumerate(MASK_COLORMAP):
            tmp_img[t == idx] = color
        color_maps.append(tmp_img.astype(np.uint8))
    return color_maps
     

def set_config():
    parse = argparse.ArgumentParser(description="RUN WITH VSX")
    parse.add_argument("--image_dir",type=str,default="/path/to/CelebAMask-HQ/test_img",help="img dir",)
    parse.add_argument("--mask_dir",type=str,default="/path/to/CelebAMask-HQ/test_label",help="mask_dir",)
    parse.add_argument("--model_prefix_path",type=str,default="deploy_weights/official_parsenet_run_stream_fp16/mod",help="model info")
    parse.add_argument("--vdsp_params_info",type=str,default="../build_in/vdsp_params/gpen-parsenet-vdsp_params.json",help="vdsp op info",)
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

    #############################################################################
    # import onnxruntime
    # session = onnxruntime.InferenceSession("deploy_weights/model_best.onnx")
    # import torch
    # device = torch.device('cpu')
    # model  = torch.jit.load("parsenet-512.torchscript.pt", map_location=device)
    # model = model.to(device)
    # model.eval()
    #############################################################################

    import sys
    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + '../../..')
    from source_code.gpen.eval.metrics import SegMetric
    classes = 19
    metrics = SegMetric(n_classes=classes)
    metrics.reset()

    input_size = [1, 3, 512, 512]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    # results = image_files
    
    for file in tqdm(image_files):
        result = vsx_inference.run_sync(file)
        #############################################################################
        image_name = os.path.basename(file)
        label_name = image_name.replace(".jpg", ".png")

        src_image = cv2.imread(file)
        image = cv2.resize(src_image, (input_size[-1], input_size[-2]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
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
        
        # # draw matting
        # load gt mask
        label = Image.open(os.path.join(args.mask_dir, label_name)).convert("L")
        label = label.resize((input_size[-1], input_size[-2]), Image.Resampling.NEAREST)
        label = np.expand_dims(label, 0)

        ori_image = Image.open(file)
        resize_image = ori_image.resize((input_size[-1], input_size[-2]), Image.Resampling.BILINEAR).convert('RGB')

        # draw image refences to source code
        vacc_pred = torch.from_numpy(result).unsqueeze(0)
        
        vacc_mask = tenor2mask(vacc_pred)[0]
        cv2.imwrite(os.path.join(args.save_dir, os.path.basename(image_name + "_vacc.png")), vacc_mask[:,:,::-1])

        # to eval
        pred = vacc_pred.data.max(1)[1].cpu().numpy()
        # eval metrics
        try:
            metrics.update(label, pred)
        except:
            continue

    score = metrics.get_scores()[0]
    class_iou = metrics.get_scores()[1]

    print("----------------- Total Performance --------------------")
    for k, v in score.items():
        print(k, v)

    print("----------------- Class IoU Performance ----------------")
    facial_names = ['background', 'skin', 'nose', 'eyeglass', 'left_eye', 'right_eye', 'left_brow', 'right_brow',
                    'left_ear', 'right_ear', 'mouth', 'upper_lip', 'lower_lip', 'hair', 'hat', 'earring', 'necklace',
                    'neck', 'cloth']
    for i in range(classes):
        print(facial_names[i] + "\t: {}".format(str(class_iou[i])))
    print("--------------------------------------------------------")

    vsx_inference.finish()
    