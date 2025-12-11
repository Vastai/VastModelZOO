# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import argparse
import glob
import os
import sys
import time
import torch
import vaststreamx as vsx
import cv2
import numpy as np
from threading import Thread, Event
import json
import cv2 as cv
from queue import Queue

from tqdm.contrib import tzip
from typing import Dict, Generator, Iterable, List, Union

_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../../..')

from source_code.CenterNet.src.lib.utils.image import get_affine_transform
from source_code.CenterNet.src.lib.models.decode import ctdet_decode
from source_code.CenterNet.src.lib.utils.post_process import ctdet_post_process
from source_code.CenterNet.src.lib.external.nms import soft_nms

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

def preprocess(image, scale=1.0):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    # official
    mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)
    # mmdet
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)
    
    if True:
      inp_height, inp_width = 512, 512
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | 31) + 1
      inp_width = (new_width | 31) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    
    # official
    # inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)
    # mmdet
    inp_image = ((inp_image - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    images = torch.from_numpy(images)
    return images.numpy(), c, s

def postprocess(dets, c, s):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [c], [s],
        128, 128, 80)
    for j in range(1, 80 + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= 1.0
    return dets[0]

def merge_outputs(detections):
    results = {}
    for j in range(1, 80 + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if True:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, 80 + 1)])
    if len(scores) > 100:
      kth = len(scores) - 100
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, 80 + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

def save_result(image_path, result, save_dir, draw=False):
    os.makedirs(save_dir, exist_ok=True)
    
    origin_img = cv2.imread(image_path)

    COLORS = np.random.uniform(0, 255, size=(200, 3))
    
    save_path = os.path.join(save_dir, os.path.basename(image_path).replace(".jpg", ".txt"))
    
    with open(save_path, 'w') as ff:
        for k, v in result.items():
            for box in v:
                '''if box[-1] > 0.5:
                    output.append(box)'''
                cls = class_names[k-1]
                bb = [cls, box[-1], box[0], box[1], box[2], box[3]]
                bb = ' '.join([str(b) for b in bb])
                ff.writelines(bb + '\n')
                
                if draw:
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    cv2.rectangle(origin_img, p1, p2, COLORS[k-1], thickness=1, lineType=cv2.LINE_AA)
                    text = f"{cls}: {round(box[-1] * 100, 2)}%"
                    y = int(int(box[1])) - 15 if int(int(box[1])) - 15 > 15 else int(int(box[1])) + 15
                    cv2.putText(
                        origin_img,
                        text,
                        (int(box[0]), y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[k-1],
                        1,
                    )
        if draw:
            file_name = os.path.split(image_path)[-1]
            cv2.imwrite(os.path.join(save_dir, file_name), origin_img)

class Detector:
    def __init__(self,
        model_prefix_path: Union[str, Dict[str, str]],
        vdsp_params_info: str,
        classes: Union[str, List[str]],
        device_id: int = 0,
        batch_size: int = 1,
        balance_mode: int = 0,
        is_async_infer: bool = False,
        model_output_op_name: str = "", ) -> None:


        if isinstance(vdsp_params_info, str):
            with open(vdsp_params_info) as f:
                vdsp_params_info_dict = json.load(f)

        if isinstance(classes, str):
            with open(classes) as f:
                classes = [cls.strip() for cls in f.readlines()]
        
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
        self.graph.add_operators(self.fusion_op, self.model_op)#

        # 构建stream
        self.infer_stream = vsx.Stream(self.graph, vsx.StreamBalanceMode.ONCE)
        if is_async_infer and len(self.model_output_op_name) > 0:
            self.infer_stream.register_operator_output(self.model_output_op_name, self.model_op)
        else:
            self.infer_stream.register_operator_output(self.model_op)
        
        self.infer_stream.build()
        
        self.classes = classes
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
                    self.post_processing(self.classes, input_id, height, width, model_output_list)
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break
    
    def finish(self,):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()

    def post_processing(self, class_list, input_id, height, width, stream_output_list):
        return self.result_dict[input_id].append(
            {
                "feature_map": stream_output_list
            },
        )

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
        model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in output[0]] ]
        result = model_output_list

        # postprocess
        origin_img = cv2.imread(image_path)

        # post processing
        yolo1_layer = np.reshape(result[0][0].astype(np.float32), result[1][0])
        yolo2_layer = np.reshape(result[0][1].astype(np.float32), result[1][1])
        yolo3_layer = np.reshape(result[0][2].astype(np.float32), result[1][2])
        
        yolo1_layer = torch.Tensor(yolo1_layer)
        yolo2_layer = torch.Tensor(yolo2_layer)
        yolo3_layer = torch.Tensor(yolo3_layer)

        hm = yolo1_layer.sigmoid_()
        wh = yolo2_layer
        reg = yolo3_layer

        detections = []
        height, width, _ = origin_img.shape

        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=False, K=100)
        dets = postprocess(dets, c, s)
        detections.append(dets)
        detections = merge_outputs(detections)

        return detections

parser = argparse.ArgumentParser(description="RUN Det WITH VSX")
parser.add_argument(
    "--file_path",
    type=str,
    default= "/path/to/det_coco_val/",
    help="img or dir  path",
)
parser.add_argument("--model_prefix_path", type=str, default="deploy_weights/official_centernet_run_stream_int8/mod", help="model info")
parser.add_argument(
    "--vdsp_params_info",
    type=str,
    default="../build_in/vsdp_params/official-centernet_res18-vdsp_params.json", 
    help="vdsp op info",
)
parser.add_argument(
    "--label_txt", type=str, default="./models/coco.txt", help="label txt"
)
parser.add_argument("--device_id", type=int, default=0, help="device id")
parser.add_argument("--batch", type=int, default=1, help="bacth size")
parser.add_argument("--save_dir", type=str, default="./output/", help="save_dir")
args = parser.parse_args()


if __name__ == "__main__":
    detector = Detector(model_prefix_path=args.model_prefix_path,
                        vdsp_params_info=args.vdsp_params_info,
                        classes=args.label_txt,
                        device_id=args.device_id,
                        batch_size=args.batch,
                        balance_mode = 0,
                        is_async_infer = False,
                        model_output_op_name = "", 
                    )

    # Test multiple images
    image_files = glob.glob(os.path.join(args.file_path + "/*.jpg"))
    time_begin = time.time()

    results = detector.run_batch(image_files)
    for (image_path, result) in tzip(image_files, results):
        # print(f"{image} => {result}")
        origin_img = cv2.imread(image_path)

        # post processing
        feature_map = result[0]["feature_map"]
        print(feature_map[0][0].shape)
        yolo1_layer = np.expand_dims(feature_map[0][0].astype(np.float32), axis=0)
        yolo2_layer = np.expand_dims(feature_map[0][1].astype(np.float32), axis=0)
        yolo3_layer = np.expand_dims(feature_map[0][2].astype(np.float32), axis=0)
        
        yolo1_layer = torch.Tensor(yolo1_layer)
        yolo2_layer = torch.Tensor(yolo2_layer)
        yolo3_layer = torch.Tensor(yolo3_layer)

        hm = yolo1_layer.sigmoid_()
        wh = yolo2_layer
        reg = yolo3_layer

        detections = []
        height, width, _ = origin_img.shape

        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=False, K=100)
        dets = postprocess(dets, c, s)
        detections.append(dets)
        detections = merge_outputs(detections)

        # save text and draw boxes
        save_result(image_path, detections, args.save_dir, draw=False)


    time_end = time.time()
    print(
        f"\n{len(image_files)} images in {time_end - time_begin} seconds, ({len(image_files) / (time_end - time_begin)} images/second)\n"
    )
    detector.finish()
    print("test over")

