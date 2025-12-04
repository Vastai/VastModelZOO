# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

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
import torch
import torch.nn as nn
import torch.nn.functional as F

parse = argparse.ArgumentParser(description="RUN Det WITH VSX")
parse.add_argument(
    "--file_path",
    type=str,
    default= "/path/to/ctw1500/test/text_images",
    help="img or dir  path",
)
parse.add_argument("--model_prefix_path", type=str, default="deploy_weights/pytorch_fast_run_stream_fp16/mod", help="model info")
parse.add_argument(
    "--vdsp_params_info",
    type=str,
    default="../build_in/vdsp_params/pytorch-fast_tiny_ctw_512_finetune_ic17mlt-vdsp_params.json", 
    help="vdsp op info",
)
parse.add_argument("--save_dir", type = str, default = "./output", help = "save_dir")
parse.add_argument("--device_id", type=int, default=0, help="device id")
parse.add_argument("--batch", type=int, default=1, help="bacth size")
args = parse.parse_args()

class FASTDetector:
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


def generate_bbox(keys, label, score, scales):
    label_num = len(keys)
    bboxes = []
    scores = []
    for index in range(1, label_num):
        i = keys[index]
        ind = (label == i)
        ind_np = ind.data.cpu().numpy()
        points = np.array(np.where(ind_np)).transpose((1, 0))
        if points.shape[0] < 200:
            label[ind] = 0
            continue
        score_i = score[ind].mean().item()
        if score_i < 0.88:
            label[ind] = 0
            continue

        # if cfg.test_cfg.bbox_type == 'rect':
        #     rect = cv2.minAreaRect(points[:, ::-1])
        #     alpha = math.sqrt(math.sqrt(points.shape[0] / (rect[1][0] * rect[1][1])))
        #     rect = (rect[0], (rect[1][0] * alpha, rect[1][1] * alpha), rect[2])
        #     bbox = cv2.boxPoints(rect) * scales

        # elif cfg.test_cfg.bbox_type == 'poly':
        binary = np.zeros(label.shape, dtype='uint8')
        binary[ind_np] = 1
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        bbox = contours[0] * scales
        bbox = bbox.astype('int32')
        bboxes.append(bbox.reshape(-1).tolist())
        scores.append(score_i)
    return bboxes, scores

def get_results(out, ori_h, ori_w, scale=2):

    out = torch.Tensor(out)

    pooling_2s = nn.MaxPool2d(kernel_size=9//2+1, stride=1, padding=(9//2) // 2)

    org_img_size = [ori_h, ori_w]
    img_size = [512, 512]  # 640*640
    batch_size = out.size(0)

    texts = F.interpolate(out[:, 0:1, :, :], size=(img_size[0] // scale, img_size[1] // scale),
                            mode='nearest')  # B*1*320*320
    texts = pooling_2s(texts)  # B*1*320*320
    score_maps = torch.sigmoid_(texts)  # B*1*320*320
    score_maps = F.interpolate(score_maps, size=(img_size[0], img_size[1]), mode='nearest')  # B*1*640*640
    score_maps = score_maps.squeeze(1)  # B*640*640
    
    kernels = (out[:, 0, :, :] > 0).to(torch.uint8)  # B*160*160
    labels_ = []
    for kernel in kernels.numpy():
        ret, label_ = cv.connectedComponents(kernel)
        labels_.append(label_)
    labels_ = np.array(labels_)
    labels_ = torch.from_numpy(labels_)
    labels = labels_.unsqueeze(1).to(torch.float32)  # B*1*160*160
    labels = F.interpolate(labels, size=(img_size[0] // scale, img_size[1] // scale), mode='nearest')  # B*1*320*320
    labels = pooling_2s(labels)
    labels = F.interpolate(labels, size=(img_size[0], img_size[1]), mode='nearest')  # B*1*640*640
    labels = labels.squeeze(1).to(torch.int32)  # B*640*640

    keys = [torch.unique(labels_[i], sorted=True) for i in range(batch_size)]


    scales = (float(org_img_size[1]) / float(img_size[1]),
                float(org_img_size[0]) / float(img_size[0]))

    results = []
    for i in range(batch_size):
        bboxes, scores = generate_bbox(keys[i], labels[i], score_maps[i], scales)
        results.append(dict(
            bboxes=bboxes,
            scores=scores
        ))

    return results

if __name__ == '__main__':
    os.makedirs(args.save_dir, exist_ok= True)
    detector = FASTDetector(model_prefix_path=args.model_prefix_path,
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
            result = get_results(result[0], ori_image.shape[0], ori_image.shape[1])
            # print(result)
            f = open(os.path.join(args.save_dir , image.split('/')[-1].split('.')[0] + '.txt'), 'w')
            for i in range(len(result)):
                box = result[i]['bboxes']
                for b in box:
                    b = [str(bb) for bb in b]
                    f.writelines(' '.join(b) + '\n')
            
            f.close()
        
        time_end = time.time()

        print(
            f"\n{len(images)} images in {time_end - time_begin} seconds, ({len(images) / (time_end - time_begin)} images/second)\n"
        )
    detector.finish()
    print("test over")
