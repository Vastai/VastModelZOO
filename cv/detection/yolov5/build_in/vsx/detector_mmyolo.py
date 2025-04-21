# ==============================================================================
#
# Copyright (C) 2024 VastaiTech Technologies Inc.  All rights reserved.
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
import torch

import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../..')
from source_code.util import non_max_suppression, scale_coords, letterbox, plot_one_box, names, colors

parse = argparse.ArgumentParser(description="RUN Det WITH VSX")
parse.add_argument(
    "--file_path",
    type=str,
    default= "/path/to/data/det_coco_val/",
    help="img or dir  path",
)
parse.add_argument("--model_prefix_path", type=str, default="./models/deploy/yolov5s-int8-max-1_3_640_640-vacc-pipeline/yolov5s", help="model info")
parse.add_argument(
    "--vdsp_params_info",
    type=str,
    default="./models/params_info/ultralytics-yolov5s-vdsp_params.json", 
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
        self.balance_mode = {0:vsx.StreamBalanceMode.ONCE, 1:vsx.StreamBalanceMode.RUN}
        self.stride = [8, 16, 32]
        self.anchor = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

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
                    input_id,height, width = self.input_dict[self.current_id]
                    model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in result[0]] ]
                    self.result_dict[input_id] = []
                    self.post_processing(self.classes, input_id, height, width, self.anchor, self.stride, model_output_list)
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break
    
    def finish(self,):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()

    def post_processing(self, class_list, input_id, height, width, anchor, stride, stream_output_list):
        z = []
        anchor = torch.tensor(anchor).view(3, 1, 3, 1, 1, 2)  # .cuda()

        result = stream_output_list[0]
        for i in range(3):
            out = result[i]
            out = torch.from_numpy(out).unsqueeze(0)  
            bs, _, ny, nx = out.shape
            grid = _make_grid(nx, ny)     # .cuda()
            out = out.view(1, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            y = out.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor[i]  # wh
            z.append(y.view(1, -1, 85))

        o = torch.cat(z, 1)
        pred = non_max_suppression(o, 0.001, 0.65, None, False, max_det=300)
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords([640, 640], det[:, :4], (height, width))  # .round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c]
                    bb = [float(x) for x in xyxy]
                    #bb = ' '.join([str(b) for b in bb])
                    score = float(conf)
                    self.result_dict[input_id].append(
                        {
                            "category_id": int(c),
                            "label": label,
                            "bbox": bb,
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
        
        nv12_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device)
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
        
        nv12_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device)
        yuv_nv12 = nv12_image#vsx.cvtcolor(nv12_image, vsx.ImageFormat.YUV_NV12)
        # bgr_inter = vsx.create_image(cv_image, vsx.ImageFormat.BGR_INTERLEAVE, cv_image.shape[1], cv_image.shape[0], self.device)
        # yuv_nv12 = vsx.cvtcolor(bgr_inter, vsx.ImageFormat.RGB_PLANAR, vsx.ImageColorSpace.COLOR_SPACE_BT709)
        
        output = self.infer_stream.run_sync([yuv_nv12])
        model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in output[0]] ]
        result = []
        # post processing
        z = []
        anchor = torch.tensor(self.anchor).view(3, 1, 3, 1, 1, 2)  # .cuda()

        result = model_output_list[0]
        for i in range(3):
            out = result[i]
            bs, _, ny, nx = out.shape
            grid = _make_grid(nx, ny)     # .cuda()
            out = out.view(1, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            y = out.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor[i]  # wh
            z.append(y.view(1, -1, 85))

        o = torch.cat(z, 1)
        pred = non_max_suppression(o, 0.001, 0.65, None, False, max_det=300)

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords([640, 640], det[:, :4], (height, width))  # .round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c]
                    bb = [label, float(conf)] + [float(x) for x in xyxy]
                    bb = ' '.join([str(b) for b in bb])
                    result.append(
                        {
                            "category_id": int(c),
                            "label": label,
                            "bbox": bb.tolist(),
                            "score": conf,
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
        print(f"{args.file_path} => {result}")
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
    
