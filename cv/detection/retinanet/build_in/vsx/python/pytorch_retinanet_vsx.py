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
import torch
import os
import cv2 as cv
from typing import Dict, Generator, Iterable, List, Union
import json
from threading import Thread, Event
import time
from queue import Queue
from tqdm import tqdm

import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../../..')

from source_code.pytorch.anchors import Anchors
from source_code.pytorch.utils import BBoxTransform, ClipBoxes
from source_code.pytorch.util import non_max_suppression, scale_coords

parse = argparse.ArgumentParser(description="RUN Det WITH VSX")
parse.add_argument(
    "--file_path",
    type=str,
    default="./work/VAMC_TEST/data/det_coco_val/",
    help="img or dir  path",
)
parse.add_argument("--model_prefix_path", type=str,
                   default="./models/deploy/yolov5s-int8-max-1_3_640_640-vacc-pipeline/yolov5s", help="model info")
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
parse.add_argument("--save_dir", type=str,
                   default="./output/", help="save_dir")
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
        cv.rectangle(origin_img, p1, p2,
                     COLORS[box["category_id"]], thickness=1, lineType=cv.LINE_AA)
        text = f"{box['label']}: {round(box['score'] * 100, 2)}%"
        y = int(int(box["bbox"][1])) - 15 if int(int(box["bbox"]
                                                     [1])) - 15 > 15 else int(int(box["bbox"][1])) + 15
        cv.putText(
            origin_img,
            text,
            (int(box["bbox"][0]), y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLORS[box["category_id"]],
            1,
        )
        print(f"{box['label']} {box['score']} {box['bbox'][0]} {box['bbox'][1]} {box['bbox'][2]+box['bbox'][0]} {box['bbox'][3]+ box['bbox'][1]}")
        fin.write(
            f"{box['label']} {box['score']} {box['bbox'][0]} {box['bbox'][1]} {box['bbox'][2]+box['bbox'][0]} {box['bbox'][3]+ box['bbox'][1]}\n")
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

        self.attr = vsx.AttrKey
        # self.device = vsx.set_device(self.device_id)
        assert vsx.set_device(self.device_id)==0
        # 构建model，模型三件套目录
        model_path = model_prefix_path
        self.model = vsx.Model(model_path, batch_size)
        # 输入预处理op
        # RGB_PLANAR
        self.fusion_op = vsx.Operator.load_ops_from_json_file(vdsp_params_info)[
            0]

        # 构建graph
        self.graph = vsx.Graph()
        self.model_op = vsx.ModelOperator(self.model)
        self.graph.add_operators(self.fusion_op, self.model_op)

        # 构建stream
        self.infer_stream = vsx.Stream(self.graph, vsx.StreamBalanceMode.ONCE)
        if is_async_infer and len(self.model_output_op_name) > 0:
            self.infer_stream.register_operator_output(
                self.model_output_op_name, self.model_op)
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
                    result = self.infer_stream.get_operator_output(
                        self.model_output_op_name)
                else:
                    result = self.infer_stream.get_operator_output(
                        self.model_op)
                if result is not None:
                    # 输出顺序和输入一致
                    self.current_id += 1
                    input_id, height, width = self.input_dict[self.current_id]
                    model_output_list = [
                        [vsx.as_numpy(out)[0].astype(np.float32) for out in result[0]]]
                    self.result_dict[input_id] = []
                    self.post_processing(
                        self.classes, input_id, height, width, model_output_list)
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
        img = np.ones((1, 3, self.model_size, self.model_size), dtype=np.uint8)

        yolo1_layer = torch.Tensor(stream_ouput_data[5]).unsqueeze(0)
        yolo2_layer = torch.Tensor(stream_ouput_data[6]).unsqueeze(0)
        yolo3_layer = torch.Tensor(stream_ouput_data[7]).unsqueeze(0)
        yolo4_layer = torch.Tensor(stream_ouput_data[8]).unsqueeze(0)
        yolo5_layer = torch.Tensor(stream_ouput_data[9]).unsqueeze(0)
        yolo6_layer = torch.Tensor(stream_ouput_data[0]).unsqueeze(0)
        yolo7_layer = torch.Tensor(stream_ouput_data[1]).unsqueeze(0)
        yolo8_layer = torch.Tensor(stream_ouput_data[2]).unsqueeze(0)
        yolo9_layer = torch.Tensor(stream_ouput_data[3]).unsqueeze(0)
        yolo10_layer = torch.Tensor(stream_ouput_data[4]).unsqueeze(0)
        print(yolo1_layer.shape)
        print(yolo2_layer.shape)
        print(yolo3_layer.shape)
        print(yolo4_layer.shape)
        print(yolo5_layer.shape)
        print(yolo6_layer.shape)
        print(yolo7_layer.shape)
        print(yolo8_layer.shape)
        print(yolo9_layer.shape)
        print(yolo10_layer.shape)

        yolo1_layer = torch.Tensor(yolo1_layer).permute(0, 2, 3, 1).view(1, int(self.model_size/8), int(self.model_size/8), 9, 80).contiguous().view(1, -1, 80)
        yolo2_layer = torch.Tensor(yolo2_layer).permute(0, 2, 3, 1).view(1, int(self.model_size/16), int(self.model_size/16), 9, 80).contiguous().view(1, -1, 80)
        yolo3_layer = torch.Tensor(yolo3_layer).permute(0, 2, 3, 1).view(1, int(self.model_size/32), int(self.model_size/32), 9, 80).contiguous().view(1, -1, 80)
        yolo4_layer = torch.Tensor(yolo4_layer).permute(0, 2, 3, 1).view(1, int(self.model_size/64), int(self.model_size/64), 9, 80).contiguous().view(1, -1, 80)
        yolo5_layer = torch.Tensor(yolo5_layer).permute(0, 2, 3, 1).view(1, int(self.model_size/128), int(self.model_size/128), 9, 80).contiguous().view(1, -1, 80)
        yolo6_layer = torch.Tensor(yolo6_layer).permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
        yolo7_layer = torch.Tensor(yolo7_layer).permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
        yolo8_layer = torch.Tensor(yolo8_layer).permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
        yolo9_layer = torch.Tensor(yolo9_layer).permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
        yolo10_layer = torch.Tensor(yolo10_layer).permute(0, 2, 3, 1).contiguous().view(1, -1, 4)

        regression = torch.cat([yolo6_layer, yolo7_layer, yolo8_layer, yolo9_layer, yolo10_layer], dim=1)
        classification = torch.cat([yolo1_layer, yolo2_layer, yolo3_layer, yolo4_layer, yolo5_layer], dim=1).sigmoid()

        boxtrans = BBoxTransform()
        clipbox = ClipBoxes()
        anchors = Anchors()(img)
        # print(anchors.shape)

        transformed_anchors = boxtrans(anchors, regression)
        transformed_anchors = clipbox(transformed_anchors, img)

        outputs = torch.cat([
            transformed_anchors,
            torch.ones((1, transformed_anchors.shape[1], 1)),
            classification
        ], dim=-1)

        pred = non_max_suppression(outputs, 0.001, 0.65, None, False, max_det=300)[0]
        if len(pred):
            pred[:, :4] = scale_coords(
                [self.model_size, self.model_size], pred[:, :4], [height, width, 3]).round()

        out = pred.numpy().tolist()
        for box in out:

            box[2] = box[2] - box[0]  # width = xmax - xmin
            box[3] = box[3] - box[1]  # height = ymax - ymin

            self.result_dict[input_id].append(
                {
                    "category_id": int(box[5]),
                    "label": self.classes[int(box[5])],
                    "bbox": box[:4],
                    "score": float(box[4]),
                }
            )

    def _run(self, image: Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv.imread(image, cv.IMREAD_COLOR)
            if cv_image.shape[0] % 2 != 0 or cv_image.shape[1] % 2 != 0:
                cv_image = cv.resize(
                    cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            image = np.stack(cv.split(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3

        nv12_image = vsx.create_image(
            image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)
        # vsx.cvtcolor(nv12_image, vsx.ImageFormat.YUV_NV12)
        yuv_nv12 = nv12_image

        input_id = self.input_id
        self.input_dict[input_id] = (input_id, height, width)
        self.event_dict[input_id] = Event()

        self.infer_stream.run_async([yuv_nv12])

        self.input_id += 1

        return input_id

    def run(self, image: Union[str, np.ndarray]):
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

    def run_sync(self, image: Union[str, np.ndarray]):
        if isinstance(image, str):
            # ori_image = cv.imread(image, cv.IMREAD_COLOR)
            cv_image = cv.imread(image, cv.IMREAD_COLOR)
            if cv_image.shape[0] % 2 != 0 or cv_image.shape[1] % 2 != 0:
                cv_image = cv.resize(
                    cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            image = np.stack(cv.split(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3

        nv12_image = vsx.create_image(
            image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)
        # vsx.cvtcolor(nv12_image, vsx.ImageFormat.YUV_NV12)
        yuv_nv12 = nv12_image
        # bgr_inter = vsx.create_image(cv_image, vsx.ImageFormat.BGR_INTERLEAVE, cv_image.shape[1], cv_image.shape[0], self.device_id)
        # yuv_nv12 = vsx.cvtcolor(bgr_inter, vsx.ImageFormat.RGB_PLANAR, vsx.ImageColorSpace.COLOR_SPACE_BT709)

        output = self.infer_stream.run_sync([yuv_nv12])
        model_output_list = [
            [vsx.as_numpy(out)[0].astype(np.float32) for out in output[0]]]
        result = []

        # postprocess
        FROM_PYTORCH = True
        stream_ouput_data = model_output_list[0]
        img = np.ones((1, 3, self.model_size, self.model_size), dtype=np.uint8)

        yolo1_layer = torch.Tensor(stream_ouput_data[5])
        yolo2_layer = torch.Tensor(stream_ouput_data[6])
        yolo3_layer = torch.Tensor(stream_ouput_data[7])
        yolo4_layer = torch.Tensor(stream_ouput_data[8])
        yolo5_layer = torch.Tensor(stream_ouput_data[9])
        yolo6_layer = torch.Tensor(stream_ouput_data[0])
        yolo7_layer = torch.Tensor(stream_ouput_data[1])
        yolo8_layer = torch.Tensor(stream_ouput_data[2])
        yolo9_layer = torch.Tensor(stream_ouput_data[3]) 
        yolo10_layer = torch.Tensor(stream_ouput_data[4])

        yolo1_layer = torch.Tensor(yolo1_layer).permute(0, 2, 3, 1).view(1, int(self.model_size/8), int(self.model_size/8), 9, 80).contiguous().view(1, -1, 80)
        yolo2_layer = torch.Tensor(yolo2_layer).permute(0, 2, 3, 1).view(1, int(self.model_size/16), int(self.model_size/16), 9, 80).contiguous().view(1, -1, 80)
        yolo3_layer = torch.Tensor(yolo3_layer).permute(0, 2, 3, 1).view(1, int(self.model_size/32), int(self.model_size/32), 9, 80).contiguous().view(1, -1, 80)
        yolo4_layer = torch.Tensor(yolo4_layer).permute(0, 2, 3, 1).view(1, int(self.model_size/64), int(self.model_size/64), 9, 80).contiguous().view(1, -1, 80)
        yolo5_layer = torch.Tensor(yolo5_layer).permute(0, 2, 3, 1).view(1, int(self.model_size/128), int(self.model_size/128), 9, 80).contiguous().view(1, -1, 80)
        yolo6_layer = torch.Tensor(yolo6_layer).permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
        yolo7_layer = torch.Tensor(yolo7_layer).permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
        yolo8_layer = torch.Tensor(yolo8_layer).permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
        yolo9_layer = torch.Tensor(yolo9_layer).permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
        yolo10_layer = torch.Tensor(yolo10_layer).permute(0, 2, 3, 1).contiguous().view(1, -1, 4)

        regression = torch.cat([yolo6_layer, yolo7_layer, yolo8_layer, yolo9_layer, yolo10_layer], dim=1)
        classification = torch.cat([yolo1_layer, yolo2_layer, yolo3_layer, yolo4_layer, yolo5_layer], dim=1).sigmoid()

        boxtrans = BBoxTransform()
        clipbox = ClipBoxes()
        anchors = Anchors()(img)

        transformed_anchors = boxtrans(anchors, regression)
        transformed_anchors = clipbox(transformed_anchors, img)

        outputs = torch.cat([
            transformed_anchors,
            torch.ones((1, transformed_anchors.shape[1], 1)),
            classification
        ], dim=-1)

        pred = non_max_suppression(outputs, 0.001, 0.65, None, False, max_det=300)[0]
        if len(pred):
            pred[:, :4] = scale_coords(
                [self.model_size, self.model_size], pred[:, :4], [height, width, 3]).round()

        out = out.numpy().tolist()
        for box in out:

            self.result_dict[input_id].append(
                {
                    "category_id": int(box[5]),
                    "label": self.classes[int(box[5])],
                    "bbox": box[:4],
                    "score": float(box[4]),
                }
            )
        return result


if __name__ == '__main__':
    detector = Detector(model_prefix_path=args.model_prefix_path,
                        vdsp_params_info=args.vdsp_params_info,
                        classes=args.label_txt,
                        device_id=args.device_id,
                        batch_size=args.batch,
                        balance_mode=0,
                        is_async_infer=False,
                        model_output_op_name="",
                        )
    if os.path.isfile(args.file_path):
        result = detector.run_sync(args.file_path)  # run(args.file_path)
        print(f"{args.file_path} => {result}")
        save_result(args.file_path, result, args.save_dir)
    else:
        # Test multiple images
        images = glob.glob(os.path.join(args.file_path, "*"))
        time_begin = time.time()
        results = detector.run_batch(images)
        for (image, result) in zip(images, results):
            # print(f"{image} => {result}")
            save_result(image, result, args.save_dir)
        time_end = time.time()

        print(
            f"\n{len(images)} images in {time_end - time_begin} seconds, ({len(images) / (time_end - time_begin)} images/second)\n"
        )
    detector.finish()
    print("test over")

'''
int8 max

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.337
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.497
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.368
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.180
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.445
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.476
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.530
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.668
{'bbox_mAP': 0.337, 'bbox_mAP_50': 0.497, 'bbox_mAP_75': 0.368, 'bbox_mAP_s': 0.18, 'bbox_mAP_m': 0.374, 'bbox_mAP_l': 0.445, 'bbox_mAP_copypaste': '0.337 0.497 0.368 0.180 0.374 0.445'}
'''