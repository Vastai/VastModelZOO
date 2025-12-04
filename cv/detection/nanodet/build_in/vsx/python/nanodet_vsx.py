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
from typing import Dict, Generator, Iterable, List, Union
import json
from threading import Thread, Event
import time
from queue import Queue
from tqdm import tqdm
import cv2 
import numpy as np
import torch
import vaststreamx as vsx
import sys
import os
import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../../../source_code')
# print(_cur_file_path + os.sep + '../..')
from nanodet.nanodet_plus_head import NanoDetPlusHead
from nanodet.transform import Pipeline

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(torch.tensor(coords), img0_shape)
    return coords

def letterbox(img, new_shape=(640, 640), color=(0, 0, 0), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # NOTE
    # 此处暂时处理为方形
    # if auto:  # minimum rectangle
    #     dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # elif scaleFill:  # stretch
    #     dw, dh = 0.0, 0.0
    #     new_unpad = (new_shape[1], new_shape[0])
    #     ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


parse = argparse.ArgumentParser(description="RUN Det WITH VSX")
parse.add_argument(
    "--file_path",
    type=str,
    default="/path/to/det_coco_val/",
    help="img or dir  path",
)
parse.add_argument("--model_prefix_path", type=str,
                   default="/path/to/nanodet_model/mod", help="model info")
parse.add_argument(
    "--vdsp_params_info",
    type=str,
    default="../build_in/vdsp_params/official-nanodet_plus_m-vdsp_params.json",
    help="vdsp op info",
)
parse.add_argument(
    "--label_txt", type=str, default="/path/to/coco.txt", help="label txt"
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
    fin = open(os.path.join(save_dir, os.path.splitext(os.path.split(image_path)[-1])[0] + ".txt"), "w")
    origin_img = cv2.imread(image_path)

    for box in result:
        p1, p2 = (int(box["bbox"][0]), int(box["bbox"][1])), (
            int(box["bbox"][2]) + int(box["bbox"][0]),
            int(box["bbox"][3]) + int(box["bbox"][1]),
        )

        cv2.rectangle(origin_img, p1, p2, COLORS[box["category_id"]], thickness=1, lineType=cv2.LINE_AA)
        text = f"{box['label']}: {round(box['score'] * 100, 2)}%"
        y = int(int(box["bbox"][1])) - 15 if int(int(box["bbox"][1])) - 15 > 15 else int(int(box["bbox"][1])) + 15
        cv2.putText(
            origin_img,
            text,
            (int(box["bbox"][0]), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLORS[box["category_id"]],
            1,
        )
        fin.write(
            f"{box['label']} {box['score']} {box['bbox'][0]} {box['bbox'][1]} {box['bbox'][2]+box['bbox'][0]} {box['bbox'][3]+ box['bbox'][1]}\n"
        )
    file_name = os.path.split(image_path)[-1]
    cv2.imwrite(os.path.join(save_dir, file_name), origin_img)
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

        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']

        yolo1_layer = torch.Tensor(stream_ouput_data[0].reshape(1, 112, 52, 52))
        yolo2_layer = torch.Tensor(stream_ouput_data[1].reshape(1, 112, 26, 26))
        yolo3_layer = torch.Tensor(stream_ouput_data[2].reshape(1, 112, 13, 13))
        yolo4_layer = torch.Tensor(stream_ouput_data[3].reshape(1, 112, 7, 7))

        pipeline = Pipeline({"normalize": [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]}, False)
        head = NanoDetPlusHead(80, None, 3, strides=[8, 16, 32, 64])
        outputs = [torch.Tensor(yolo1_layer).flatten(start_dim=2), torch.Tensor(yolo2_layer).flatten(start_dim=2), torch.Tensor(yolo3_layer).flatten(start_dim=2), torch.Tensor(yolo4_layer).flatten(start_dim=2)]
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        img_info = {"id": 0}
        img_info["file_name"] = None
        img_info["height"] = [416]
        img_info["width"] = [416]

        img, _, _ = letterbox(self.image, (416, 416))
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = pipeline(None, meta, [416, 416])
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).unsqueeze(0)
        result = head.post_process(outputs, meta)[0]
        # result_img = head.show_result(meta["raw_img"], result, class_names, score_thres=0.35, show=False)

        for id, bbox in result.items():
            if len(bbox) != 0:
                for _box in bbox:
                    b = np.array(_box[:4]).reshape((1, -1))
                    b = scale_coords((416, 416), b, (height, width, 3)).round()[0]
                    bb = [b[0], b[1], b[2]-b[0], b[3]-b[1]]
                    self.result_dict[input_id].append(
                        {
                            "category_id": id,
                            "label": class_names[id],
                            "bbox": bb,
                            "score": float(_box[4]),
                        }
                    )

    def _run(self, image: Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            # cv_image = cv2.resize(cv_image, (self.model_size, self.model_size))
            self.image = cv_image.copy()
            if cv_image.shape[0] % 2 != 0 or cv_image.shape[1] % 2 != 0:
                cv_image = cv2.resize(
                    cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
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
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            if cv_image.shape[0] % 2 != 0 or cv_image.shape[1] % 2 != 0:
                cv_image = cv2.resize(
                    cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
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
        cls_score = torch.Tensor(stream_ouput_data[0])
        bbox_reg = torch.Tensor(stream_ouput_data[1])
        objectness = torch.Tensor(stream_ouput_data[2])

        anchor = AnchorGenerator([32], [1.], [1., 2, 4, 8, 16])
        all_anchors = anchor.grid_priors([(int(self.model_size/32), int(self.model_size/32))], device='cpu')

        target_means = (0., 0., 0., 0.),
        target_stds = (1., 1., 1., 1.),


        N = 1
        _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, 80, H, W)

        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(
            1. + torch.clamp(cls_score.exp(), max=1e8) +
            torch.clamp(objectness.exp(), max=1e8))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)

        cls_score_torch = normalized_cls_score[0].permute(
            1, 2, 0).reshape(-1, 80)
        bbox_pred_torch = bbox_reg.permute(1, 2, 0).reshape(-1, 4)
        cls_score_torch = cls_score_torch.sigmoid()

        pred = delta2bbox(
            all_anchors[0], bbox_pred_torch, target_means, target_stds)

        out = torch.cat((pred, torch.ones((5*int(self.model_size/32)*int(self.model_size/32), 1)), cls_score_torch), axis=1)
        out = out.unsqueeze(0)
        out = non_max_suppression(out, 0.05, 0.6)[0]
        if len(out):
            out[:, :4] = scale_coords(
                [self.model_size, self.model_size], out[:, :4], [height, width, 3]).round()

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
