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
from tqdm import tqdm

parse = argparse.ArgumentParser(description="RUN Det WITH VSX")

parse.add_argument("--file_path",
                   type=str,
                   default= "/path/to/det_coco_val/",
                   help="img or dir  path",
)

parse.add_argument("--model_prefix_path", 
                   type=str, 
                   default="/path/to/weights/mod", 
                   help="model info")

parse.add_argument("--vdsp_params_info",
                   type=str,
                   default="/path/to/vdsp_params.json", 
                   help="vdsp op info",)

parse.add_argument("--label_txt", 
                   type=str, default="/path/to/coco.txt", 
                   help="label txt")

parse.add_argument("--device_id", type=int, default=0, help="device id")
parse.add_argument("--batch", type=int, default=1, help="bacth size")
parse.add_argument("--save_dir", type=str, default="./output/", help="save_dir")
args = parse.parse_args()


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
        fin.write(f"{box['label']} {box['score']} {box['bbox'][0]} {box['bbox'][1]} {box['bbox'][2]+box['bbox'][0]} {box['bbox'][3]+ box['bbox'][1]}\n" )
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

        FROM_PYTORCH = True
        stream_ouput_data = stream_output_list[0]
        box_ids = stream_ouput_data[0]
        classes = len(box_ids)
        box_scores = stream_ouput_data[1]
        # print('box_scores:\n', box_scores)

        # (classes, 4): xmin, ymin, xmax, ymax
        box_boxes = stream_ouput_data[2]
        box_boxes = np.reshape(box_boxes, (classes, 4))
        # print('box_boxes:\n', box_boxes)

        # post processing
        r = min(self.model_size / width, self.model_size / height)
        unpad_w = int(round(width * r))
        unpad_h = int(round(height * r))
        dw = self.model_size - unpad_w
        dh = self.model_size - unpad_h
        dw /= 2
        dh /= 2
        for i in range(classes):
            if box_ids[i] == -1:
                break
            box_xy = box_boxes[i]
            # print('box_xy:\n', box_xy)
            box_xy[0] = (box_xy[0] - dw) * width / unpad_w
            box_xy[2] = (box_xy[2] - dw) * width / unpad_w
            box_xy[1] = (box_xy[1] - dh) * height / unpad_h
            box_xy[3] = (box_xy[3] - dh) * height / unpad_h

            # width, height
            if FROM_PYTORCH:
                box_xy[2] = box_xy[2] - box_xy[0]  # width = xmax - xmin
                box_xy[3] = box_xy[3] - box_xy[1]  # height = ymax - ymin
            else:
                box_xy[2] = box_xy[2] - box_xy[0] + 1  # width = xmax - xmin + 1
                box_xy[3] = box_xy[3] - box_xy[1] + 1  # height = ymax - ymin + 1

            self.result_dict[input_id].append(
                {
                    "category_id": int(box_ids[i]),
                    "label": class_list[int(box_ids[i])],
                    "bbox": box_xy[:4].tolist(),
                    "score": box_scores[i].item(),
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

        # postprocess
        FROM_PYTORCH = True
        stream_ouput_data = model_output_list[0]
        box_ids = stream_ouput_data[0]
        classes = len(box_ids)
        box_scores = stream_ouput_data[1]
        # print('box_scores:\n', box_scores)

        # (classes, 4): xmin, ymin, xmax, ymax
        box_boxes = stream_ouput_data[2]
        box_boxes = np.reshape(box_boxes, (classes, 4))
        # print('box_boxes:\n', box_boxes)

        # post processing
        r = min(self.model_size / width, self.model_size / height)
        unpad_w = int(round(width * r))
        unpad_h = int(round(height * r))
        dw = self.model_size - unpad_w
        dh = self.model_size - unpad_h
        dw /= 2
        dh /= 2
        for i in range(classes):
            if box_ids[i] == -1:
                break
            box_xy = box_boxes[i]
            # print('box_xy:\n', box_xy)
            box_xy[0] = (box_xy[0] - dw) * width / unpad_w
            box_xy[2] = (box_xy[2] - dw) * width / unpad_w
            box_xy[1] = (box_xy[1] - dh) * height / unpad_h
            box_xy[3] = (box_xy[3] - dh) * height / unpad_h

            # width, height
            if FROM_PYTORCH:
                box_xy[2] = box_xy[2] - box_xy[0]  # width = xmax - xmin
                box_xy[3] = box_xy[3] - box_xy[1]  # height = ymax - ymin
            else:
                box_xy[2] = box_xy[2] - box_xy[0] + 1  # width = xmax - xmin + 1
                box_xy[3] = box_xy[3] - box_xy[1] + 1  # height = ymax - ymin + 1

            result.append(
                {
                    "category_id": int(box_ids[i]),
                    "label": self.classes[int(box_ids[i])],
                    "bbox": box_xy[:4].tolist(),
                    "score": box_scores[i].item(),
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
                        model_output_op_name = "",)
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
    
