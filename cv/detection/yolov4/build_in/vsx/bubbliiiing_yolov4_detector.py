# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :          lance
@Email : lance.wang@vastaitech.com
@Time  : 	2025/04/21 19:43:31
'''

from curses import has_key
import vaststreamx as vsx
import cv2
import numpy as np
import argparse
import glob
import os
import cv2 as cv
from tqdm import tqdm
from typing import Dict, Generator, Iterable, List, Union
import json
from threading import Thread, Event
import time
from queue import Queue
import torch

import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../..')
from source_code.bubbliiiing import utils_bbox

parse = argparse.ArgumentParser(description="RUN Det WITH VSX")
parse.add_argument(
    "--file_path",
    type=str,
    default= "/path/to/det_coco_val/",
    help="img or dir  path",
)
parse.add_argument("--model_prefix_path", type=str, default="./models/deploy/mod", help="model info")
parse.add_argument(
    "--vdsp_params_info",
    type=str,
    default="./models/params_info/vdsp.json", 
    help="vdsp op info",
)
parse.add_argument(
    "--label_txt", type=str, default="./models/coco.txt", help="label txt"
)
parse.add_argument("--device_id", type=int, default=0, help="device id")
parse.add_argument("--batch", type=int, default=1, help="bacth size")
parse.add_argument("--save_dir", type=str, default="./output/", help="save_dir")
args = parse.parse_args()

def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def save_result(image_path, result, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    COLORS = np.random.uniform(0, 255, size=(200, 3))
    file_dir = image_path.split('/')[-2]
    save_txt = image_path.split('/')[-1].replace('jpg', 'txt')
    fin = open(os.path.join(save_dir, save_txt), "w")
    # fin.write('{:s}\n'.format('%s/%s' % (file_dir, image_path.split('/')[-1][:-4])))
    # fin.write('{:d}\n'.format(len(result)))
    origin_img = cv.imread(image_path)
    for box in result:
        if box['score'] < 0.01:
            continue
        p1, p2 = (int(box["bbox"][0]), int(box["bbox"][1])), (
            int(box["bbox"][2]) + int(box["bbox"][0]),
            int(box["bbox"][3]) + int(box["bbox"][1]),
        )
        cv.rectangle(origin_img, p1, p2, COLORS[box["category_id"]], thickness=1, lineType=cv.LINE_AA)
        text = f"{box['label']}: {round(box['score'] * 100, 2)}%"
        y = int(int(box["bbox"][1])) - 15 if int(int(box["bbox"][1])) - 15 > 15 else int(int(box["bbox"][1])) + 15
        cv.putText(
            origin_img,
            text,
            (int(box["bbox"][0]), y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLORS[box["category_id"]],
            1,
        )
        fin.write(f"{box['label']} {box['score']} {box['bbox'][0]} {box['bbox'][1]} {box['bbox'][2]} {box['bbox'][3]}\n" )
        # fin.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(box['bbox'][0], box['bbox'][1], box['bbox'][2], box['bbox'][3], box['score']))
    file_name = os.path.split(image_path)[-1]
    # cv.imwrite(os.path.join(save_dir, file_name), origin_img)
    fin.close()


class Detector:
    def __init__(self,
        model_prefix_path: Union[str, Dict[str, str]],
        vdsp_params_info: str,
        classes: Union[str, List[str]],
        device_id: int = 0,
        batch_size: int = 1,
        balance_mode: int = 0,
        is_async_infer: bool = False,
        model_output_op_name: str = "", 
        conf_thres: float = 0.01,
        nms_thres: float = 0.65) -> None:


        if isinstance(vdsp_params_info, str):
            with open(vdsp_params_info) as f:
                vdsp_params_info_dict = json.load(f)

        if isinstance(classes, str):
            with open(classes) as f:
                classes = [cls.strip() for cls in f.readlines()]
        
        self.model_size = [vdsp_params_info_dict["OpConfig"]["OimageHeight"], vdsp_params_info_dict["OpConfig"]["OimageWidth"]]
        self.device_id = device_id
        self.model_output_op_name = model_output_op_name
        self.input_id = 0
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.balance_mode = {0:vsx.StreamBalanceMode.ONCE, 1:vsx.StreamBalanceMode.RUN}
        self.stride = [8, 16, 32]
        self.anchor = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

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
        self.infer_stream = vsx.Stream(self.graph, self.balance_mode[balance_mode])
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
                    input_id, height, width = self.input_dict[self.current_id]
                    model_output_list = [np.expand_dims(vsx.as_numpy(out)[0].astype(np.float32), axis=0) for out in result[0]] 
                    self.result_dict[input_id] = []
                    self.post_processing(model_output_list, int(height), int(width), int(input_id))
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break
    
    def finish(self,):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()

    def post_processing(self, stream_ouput, height, width, input_id, draw_image=False, **kwargs):
        # post proecess
        # decode
        # 416 anchors
        anchors = [12, 16,  19, 36,  40, 28,  36, 75,  76, 55,  72, 146,  142, 110,  192, 243,  459, 401]
        if len(stream_ouput) < 3:
            # yolov4_tiny
            anchors = [10,14,  23,27,  37,58,  81,82,  135,169,  344,319]

        anchors = np.array(anchors).reshape(-1, 2)
        bbox_util = utils_bbox.DecodeBoxNP(anchors=anchors, num_classes=len(self.classes), input_shape=self.model_size)
        if len(stream_ouput) < 3:
            # yolov4_tiny
            bbox_util = utils_bbox.DecodeBoxNP(anchors=anchors, num_classes=len(self.classes), input_shape=self.model_size, anchors_mask=[[3, 4, 5], [1, 2, 3]])
        
        outputs = bbox_util.decode_box(stream_ouput)
        
        # nms
        image_shape = np.array((height, width))
        results = bbox_util.non_max_suppression(np.concatenate(outputs, 1),
                                                num_classes=len(self.classes),
                                                input_shape=self.model_size,
                                                image_shape=image_shape,
                                                letterbox_image=True,
                                                conf_thres=self.conf_thres,
                                                nms_thres=self.nms_thres)
        if(results[0] is not None):
            top_label = np.array(results[0][:, 6], dtype='int32')
            # print(top_label)
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
            res_length = len(top_boxes)
            
            # 推入异步队列
            if res_length:
                COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
                for index in range(res_length):
                    label = self.classes[top_label[index].astype(np.int8)]
                    score = top_conf[index]
                    bbox = top_boxes[index].tolist()
                    bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
                    bbox[0] = np.clip(bbox[0], 0, image_shape[1])
                    bbox[2] = np.clip(bbox[2], 0, image_shape[1])
                    bbox[1] = np.clip(bbox[1], 0, image_shape[0])
                    bbox[3] = np.clip(bbox[3], 0, image_shape[0])
                    self.result_dict[input_id].append(
                        {
                            "category_id": int(top_label[index]) ,
                            "label": label,
                            "bbox": bbox,
                            "score": score,
                        }
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
        model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in output[0]] ]
        result = []
        # post processing
        z = []
        anchor = torch.tensor(self.anchor).view(3, 1, 3, 1, 1, 2)  # .cuda()

        stream_ouput = model_output_list[0]
        anchors = [12, 16,  19, 36,  40, 28,  36, 75,  76, 55,  72, 146,  142, 110,  192, 243,  459, 401]
        if len(stream_ouput) < 3:
            # yolov4_tiny
            anchors = [10,14,  23,27,  37,58,  81,82,  135,169,  344,319]

        anchors = np.array(anchors).reshape(-1, 2)
        bbox_util = utils_bbox.DecodeBoxNP(anchors=anchors, num_classes=len(self.classes), input_shape=self.model_size)
        if len(stream_ouput) < 3:
            # yolov4_tiny
            bbox_util = utils_bbox.DecodeBoxNP(anchors=anchors, num_classes=len(self.classes), input_shape=self.model_size, anchors_mask=[[3, 4, 5], [1, 2, 3]])
        
        outputs = bbox_util.decode_box(stream_ouput)
        
        # nms
        image_shape = np.array(height, width)
        results = bbox_util.non_max_suppression(np.concatenate(outputs, 1),
                                                num_classes=len(self.classes),
                                                input_shape=self.model_size,
                                                image_shape=image_shape,
                                                letterbox_image=True,
                                                conf_thres=self.conf_thres,
                                                nms_thres=self.nms_thres)
        
        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]
        res_length = len(top_boxes)
        
        if res_length:
            COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
            for index in range(res_length):
                label = self.classes[top_label[index].astype(np.int8)]
                score = top_conf[index]
                bbox = top_boxes[index].tolist()
                bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
        
                result.append(
                        {
                            "category_id": int(top_label[index]) ,
                            "label": label,
                            "bbox": bbox,
                            "score": score,
                        }
                    )

        return result


if __name__ == '__main__':
    detector = Detector(model_prefix_path=args.model_prefix_path,
                        vdsp_params_info=args.vdsp_params_info,
                        classes=args.label_txt,
                        device_id=args.device_id,
                        batch_size=args.batch,
                        balance_mode = 0,
                        is_async_infer = False,
                        model_output_op_name = "", 
                    )
    if os.path.isfile(args.file_path):
        result = detector.run_sync(args.file_path)#run(args.file_path)
        #print(f"{args.file_path} => {result}")
        save_result(args.file_path, result, args.save_dir)
    else:
        # Test multiple images
        images = glob.glob(os.path.join(args.file_path + "/*"))
        time_begin = time.time()
        results = detector.run_batch(images)
        for (image, result) in zip(images, results):
            #print(f"{image} => {result}")
            save_result(image, result, args.save_dir)
        time_end = time.time()

        print(
            f"\n{len(images)} images in {time_end - time_begin} seconds, ({len(images) / (time_end - time_begin)} images/second)\n"
        )
    detector.finish()
    print("test over")
    
