# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from curses import has_key
import vaststreamx as vsx
import numpy as np
import argparse
import glob
import os
import cv2 
from typing import Dict, Generator, Iterable, List, Union
import json
from threading import Thread, Event
import time
from queue import Queue
from tqdm import tqdm
import torch

import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../../../source_code')

from utils import (non_max_suppression, DFL, dist2bbox, make_anchors, scale_coords, coco80_to_coco91_class, coco_names, xyxy2xywh)

parse = argparse.ArgumentParser(description="RUN Det WITH VSX")
parse.add_argument(
    "--file_path",
    type=str,
    default= "/path/to/data/det_coco_val/",
    help="img or dir  path",
)
parse.add_argument("--model_prefix_path", type=str, default="deploy_weights/ultralytics_yolov8_pose_run_stream_int8/mod", help="model info")
parse.add_argument(
    "--vdsp_params_info",
    type=str,
    default="../build_in/params_info/ultralytics-yolov8s_pose-vdsp_params.json", 
    help="vdsp op info",
)
parse.add_argument(
    "--label_txt", type=str, default="/path/to/coco.txt", help="label txt"
)
parse.add_argument("--device_id", type=int, default=0, help="device id")
parse.add_argument("--batch", type=int, default=1, help="bacth size")
parse.add_argument("--save_dir", type=str, default="./output/", help="save_dir")
args = parse.parse_args()

output_shape = [[1, 64, 80 ,80], [1, 1, 80, 80], [1, 64, 40, 40], [1, 1, 40, 40], [1, 64, 20, 20],
                [1, 1, 20, 20], [1, 51, 80, 80], [1, 51, 40, 40], [1, 51, 20, 20]]

class Segmenter:
    def __init__(
        self,
        model_size: Union[int, list],
        classes: Union[str, List[str]],
        threashold: float = 0.01
    ) -> None:
        if isinstance(classes, str):
            with open(classes) as f:
                classes = [cls.strip() for cls in f.readlines()]

        if isinstance(model_size,int):
            self.model_size = [model_size, model_size]
        else:
            # h,w
            assert len(model_size) == 2 , "model_size ERROR."
            self.model_size = model_size # h,w
        self.classes = classes
        self.threashold = threashold
        self.jdict = []

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            # gain  = old / new
            gain = min(img1_shape[0] / img0_shape[0],
                       img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / \
                2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain

        self.clip_coords(coords, img0_shape)
        return coords

    def kpts_decode(self, kpts, anchors, strides):
        ndim = 3
        y = kpts.clone()
        # if ndim == 3:
        y[:, 2::3].sigmoid_()  # inplace sigmoid
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (anchors[0] - 0.5)) * strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (anchors[1] - 0.5)) * strides
        return y

    def clip_coords(self, boxes, shape):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def postprocess(self, stream_ouput, classes_list, image_file, save_dir, save_img=False, **kwargs):
        #print(f"=====>>>>>process img-{image_file}")
        origin_img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        height, width, _ = origin_img.shape
        file_name = os.path.basename(image_file)
        
        dfl = DFL(16)
        for i in range(9):
            stream_ouput[i] = torch.Tensor(stream_ouput[i].reshape(output_shape[i]))
        
        ## detect cat
        output = []
        for i in range(3):
            x = torch.cat((stream_ouput[i*2], stream_ouput[i*2+1]), 1)
            output.append(x)

        anchors, strides = (x.transpose(0, 1) for x in make_anchors(output, [8, 16, 32], 0.5))

        x_cat = torch.cat([xi.view(1, 65, -1) for xi in output], 2)
        box, cls = x_cat.split((16 * 4, 1), 1)
        dbox = dist2bbox(dfl(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
        ty = torch.cat((dbox, cls.sigmoid()), 1)
       
        det_out = (ty, output)

        ## keypoints cat
        tkpt = []
        for i in range(6, 9):
            tkpt.append(stream_ouput[i].view(1, 51, -1))
        kpt = torch.cat(tkpt, -1)
        #print(strides)
        pkpt = self.kpts_decode(kpt, anchors, strides)
        kpt_out = (torch.cat([det_out[0], pkpt], 1), (det_out[1], kpt))
        pred = non_max_suppression(kpt_out)[0]
        # print(pred)
        npr = pred.shape[0]
        if npr != 0:
            predn = pred.clone()
            # print(f"npr-{npr}")
            pred_kpts = predn[:, 6:].view(npr, 17, -1)
            scale_r = self.model_size[0] / max(height, width)
            pad_h = (self.model_size[0] - height * scale_r) / 2
            pad_w = (self.model_size[0] - width * scale_r) / 2
            scale_coords(self.model_size, pred_kpts, (height, width), ratio_pad=((scale_r, scale_r), (pad_w, pad_h)))
            if len(pred):
                pred[:, :4] = self.scale_coords(self.model_size, pred[:, :4], [height, width, 3]).round()
            
            pred = pred.numpy()
            res_length = len(pred)
            COLORS = np.random.uniform(0, 255, size=(len(classes_list), 3))
            
            # 画框
            box_list = []
            if res_length:
                for index in range(res_length):
                    label = classes_list[pred[index][5].astype(np.int8)]
                    score = pred[index][4]
                    bbox = pred[index][:4].tolist()
                    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(origin_img, p1, p2, (0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    text = f"{label}: {round(score * 100, 2)}%"
                    y = int(int(bbox[1])) - 15 if int(int(bbox[1])) - 15 > 15 else int(int(bbox[1])) + 15
                    cv2.putText(origin_img,text,(int(bbox[0]), y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[pred[index][5].astype(np.int8)],2,)
                    box_list.append(f"{label} {score} {int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])}")
                if save_img:
                    cv2.imwrite(f"{save_dir}/{file_name}", origin_img)
            new_filename = os.path.splitext(file_name)[0]
            self.pred_to_json(box_list, new_filename,  predn)

    def pred_to_json(self, box_list, file_name,  predn):
        """Save one JSON result."""
        coco_num = coco80_to_coco91_class()
        image_id = int(file_name) if file_name.isnumeric() else file_name
        box = []
        label = []
        score = []
        for line in box_list:
            line = line.strip().split()
            label.append(coco_num[coco_names.index(" ".join(line[:-5]))])
            box.append([float(l) for l in line[-4:]])
            score.append(float(line[-5]))
        if len(box):
            box = xyxy2xywh(np.array(box))  # x1y1wh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            kptlist  = predn.tolist()

            for i in range(len(box.tolist())):
                self.jdict.append({
                    "image_id": image_id,
                    "category_id": label[i],
                    "bbox": [x for x in box[i].tolist()],
                    # 'keypoints': [x[6:] for x in predn.tolist()], 
                    'keypoints': kptlist[i][6:],
                    "score": score[i]
                    }
                )

    def save_json(self, json_save_dir):
        with open(json_save_dir+"/predictions.json", 'w') as f:
            json.dump(self.jdict, f)  # flatten and save

    def feature_decode(self, input_image_path: str, feature_maps: list, json_save_dir):
        stream_ouput = feature_maps
        # post proecess
        self.postprocess(stream_ouput, self.classes, input_image_path, json_save_dir, save_img=False)
        return stream_ouput

class PoseDetector:
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
        
        self.model_size_width = vdsp_params_info_dict["OpConfig"]["OimageWidth"]
        self.model_size_height = vdsp_params_info_dict["OpConfig"]["OimageHeight"]
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
        output_data = stream_output_list[0]
        self.result_dict[input_id].append(
            {
                "features": output_data,
            }
        )
        
    def _run(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            if cv_image.shape[0]%2 != 0 or cv_image.shape[1]%2 != 0:
                cv_image = cv2.resize(cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
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
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            if cv_image.shape[0]%2 != 0 or cv_image.shape[1]%2 != 0:
                cv_image = cv2.resize(cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
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

        # postprocessing 

        return result


if __name__ == '__main__':
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    pose_detector = PoseDetector(model_prefix_path=args.model_prefix_path,
                        vdsp_params_info=args.vdsp_params_info,
                        classes=args.label_txt,
                        device_id=args.device_id,
                        batch_size=args.batch,
                        balance_mode = 0,
                        is_async_infer = False,
                        model_output_op_name = "", 
                    )
    
    feature_decoder = Segmenter(model_size=[pose_detector.model_size_height, pose_detector.model_size_width], classes=args.label_txt, threashold=0.01)
    print(args.file_path)
    if os.path.isfile(args.file_path):
        result = pose_detector.run_sync(args.file_path)#run(args.file_path)
        print(f"{args.file_path} => {result}")
        save_result(args.file_path, result, args.save_dir)
    else:
        # Test multiple images
        images = glob.glob(os.path.join(args.file_path, "*"))
        time_begin = time.time()
        results = pose_detector.run_batch(images)
        
        for (image, result) in zip(images, results):
            basename, _ = os.path.splitext(os.path.basename(image))
            features = np.squeeze(result[0]["features"])
            # 关键点解码
            feature_decoder.feature_decode(image, features, args.save_dir)

        time_end = time.time()

        print(
            f"\n{len(images)} images in {time_end - time_begin} seconds, ({len(images) / (time_end - time_begin)} images/second)\n"
        )
    feature_decoder.save_json(args.save_dir)
    pose_detector.finish()
    print("test over")
    
