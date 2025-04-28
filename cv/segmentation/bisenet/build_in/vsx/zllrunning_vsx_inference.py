
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
import time
import json
import glob
import math
import copy
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
        self.device = vsx.set_device(self.device_id)
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
        
        input_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device)

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
        
        input_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device)

        output = self.infer_stream.run_sync([input_image])
        model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in output[0]] ]
        output_data = model_output_list[0][0]

        return output_data
    

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = copy.deepcopy(im).astype(np.uint8)
    vis_parsing_anno = copy.deepcopy(parsing_anno).astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.6, vis_parsing_anno_color, 0.4, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'_onehot.png', vis_parsing_anno)
        cv2.imwrite(save_path[:-4] +'_parsing.png', vis_parsing_anno_color)
        cv2.imwrite(save_path[:-4] +'_vis.png', vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_parsing_anno, vis_im


def set_config():
    parse = argparse.ArgumentParser(description="RUN WITH VSX")
    parse.add_argument(
        "--image_dir",
        type=str,
        default="./dataset/sr/CelebAMask-HQ/bisegnet_test_img",
        help="img dir",
    )
    parse.add_argument(
        "--mask_dir",
        type=str,
        default="./dataset/sr/CelebAMask-HQ/bisegnet_test_mask",
        help="mask_dir",
    )
    parse.add_argument(
        "--model_prefix_path",
        type=str,
        default="deploy_weights/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod",
        help="model info")
    parse.add_argument(
        "--vdsp_params_info",
        type=str,
        default="segmentation/bisenet/vacc_code/vdsp_params/zllrunning-bisenet-vdsp_params.json",
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
    image_files = glob.glob(os.path.join(args.image_dir, "*.jpg"))

    os.makedirs(args.save_dir, exist_ok=True)

    results = vsx_inference.run_batch(image_files)
    #############################################################################
    # import onnxruntime
    # session = onnxruntime.InferenceSession("deploy_weights/model_best.onnx")
    # import torch
    # device = torch.device('cpu')
    # model  = torch.jit.load("bisenet-512.torchscript.pt", map_location=device)
    # model = model.to(device)
    # model.eval()
    #############################################################################
    import sys
    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + '../..')
    from source_code.face_parsing.eval.metrics import SegMetric
    
    classes = 19
    metrics = SegMetric(n_classes=classes)
    metrics.reset()
    input_size = [1, 3, 512, 512]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # results = image_files
    for (image_path, result) in tzip(image_files, results):
        #############################################################################
        image_name = os.path.basename(image_path)
        label_name = image_name.replace(".jpg", ".png")

        src_image = cv2.imread(image_path)
        image = cv2.resize(src_image, (input_size[3], input_size[2]))
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

        # load gt mask
        label = Image.open(os.path.join(args.mask_dir, label_name)).convert("L")
        label = label.resize((input_size[3], input_size[2]), Image.Resampling.NEAREST)
        label = np.expand_dims(label, 0).astype("int64")

        ori_image = Image.open(image_path)
        resize_image = ori_image.resize((input_size[3], input_size[2]), Image.Resampling.BILINEAR).convert('RGB')

        heatmap = np.expand_dims(result, axis=0)
        # print(image_path, "--->", result)
        
        # # draw image refences to source code
        vacc_pred = torch.from_numpy(heatmap)

        parsing = vacc_pred.squeeze(0).cpu().numpy().argmax(0)
        vis_parsing_maps(resize_image, parsing, stride=1, save_im=True, save_path=os.path.join(args.save_dir, os.path.basename(image_path)))

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
    

"""
deploy_weights/bisenet-int8-kl_divergence-3_512_512-vacc

----------------- Total Performance --------------------
Overall Acc:     0.956146761203189
Mean Acc :       0.8349050720972586
FreqW Acc :      0.9174659778428605
Mean IoU :       0.7439906275016586
Overall F1:      0.8419505286229952
----------------- Class IoU Performance ----------------
background      : 0.9385468927289558
skin    : 0.9286480447169513
nose    : 0.6142816733363917
eyeglass        : 0.6068330309489531
left_eye        : 0.6483844965944279
right_eye       : 0.6407434575270676
left_brow       : 0.8360630615674347
right_brow      : 0.6716693847321674
left_ear        : 0.6557752923469459
right_ear       : 0.42048665436797983
mouth   : 0.8703987116305426
upper_lip       : 0.8361840308212896
lower_lip       : 0.7624259885445123
hair    : 0.7992178920817571
hat     : 0.8632830611012826
earring : 0.3614878430833457
necklace        : 0.8412396234444498
neck    : 0.9304980095981971
cloth   : 0.9096547733588586
--------------------------------------------------------

deploy_weights/bisenet-fp16-none-3_512_512-vacc

----------------- Total Performance --------------------
Overall Acc:     0.9563868560676202
Mean Acc :       0.836159020625529
FreqW Acc :      0.9178678710377364
Mean IoU :       0.7457074510478688
Overall F1:      0.8432098503134774
----------------- Class IoU Performance ----------------
background      : 0.9389768802916137
skin    : 0.928843339224967
nose    : 0.6210202184173248
eyeglass        : 0.6082734379727116
left_eye        : 0.6528055614899406
right_eye       : 0.6491106122588511
left_brow       : 0.836329468362994
right_brow      : 0.6747255043096295
left_ear        : 0.6612658198890043
right_ear       : 0.41995398544876844
mouth   : 0.8709640057588639
upper_lip       : 0.8365236325894498
lower_lip       : 0.7629834700366042
hair    : 0.799893848984409
hat     : 0.8633119827979584
earring : 0.36243221331411585
necklace        : 0.8403389293906516
neck    : 0.9309202899546812
cloth   : 0.9097683694169695
--------------------------------------------------------
"""
