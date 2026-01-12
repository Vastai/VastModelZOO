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
import cv2 as cv
import math
from typing import Dict, Generator, Iterable, List, Union
import json
from threading import Thread, Event
import time
from queue import Queue
from shapely.geometry import Polygon

# YOLOv12-OBB的类别
CLASSES = ['plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 
          'basketball court', 'ground track field', 'harbor', 'bridge', 'large vehicle', 
          'small vehicle', 'helicopter', 'roundabout', 'soccer ball field', 'swimming pool']

nmsThresh = 0.4
objectThresh = 0.5

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, angle):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.angle = angle

def rotate_rectangle(x1, y1, x2, y2, a):
    # 计算中心点坐标
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # 对每个顶点进行旋转变换
    x1_new = int((x1 - cx) * math.cos(a) - (y1 - cy) * math.sin(a) + cx)
    y1_new = int((x1 - cx) * math.sin(a) + (y1 - cy) * math.cos(a) + cy)

    x2_new = int((x2 - cx) * math.cos(a) - (y2 - cy) * math.sin(a) + cx)
    y2_new = int((x2 - cx) * math.sin(a) + (y2 - cy) * math.cos(a) + cy)

    x3_new = int((x1 - cx) * math.cos(a) - (y2 - cy) * math.sin(a) + cx)
    y3_new = int((x1 - cx) * math.sin(a) + (y2 - cy) * math.cos(a) + cy)

    x4_new = int((x2 - cx) * math.cos(a) - (y1 - cy) * math.sin(a) + cx)
    y4_new = int((x2 - cx) * math.sin(a) + (y1 - cy) * math.cos(a) + cy)
    return [(x1_new, y1_new), (x3_new, y3_new), (x2_new, y2_new), (x4_new, y4_new)]

def intersection(g, p):
    g = np.asarray(g)
    p = np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union

def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)
    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId
        angle = sort_detectboxs[i].angle
        p1 = rotate_rectangle(xmin1, ymin1, xmax1, ymax1, angle)
        p1 = np.array(p1).reshape(-1)
        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    angle2 = sort_detectboxs[j].angle
                    p2 = rotate_rectangle(xmin2, ymin2, xmax2, ymax2, angle2)
                    p2 = np.array(p2).reshape(-1)
                    iou = intersection(p1, p2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    # 将输入向量减去最大值以提高数值稳定性
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def process_yolov12_obb_output(outputs_model, input_size=640):
    """
    处理YOLOv12-OBB模型的输出
    """
    outputs_new = []
    for i in range(len(outputs_model)):
        outputs_new.append(np.expand_dims(outputs_model[i], axis=0))

    # 将前6个输出两两concat (0-1, 2-3, 4-5)
    concat_outputs = []
    for i in range(0, 6, 2):
        if i + 1 < len(outputs_new):
            concat_out = np.concatenate([outputs_new[i], outputs_new[i+1]], axis=1)
            concat_outputs.append(concat_out)

    # 添加角度输出（第6,7,8个）
    angle_outputs = []
    for i in range(6, len(outputs_new)):
        angle_outputs.append(outputs_new[i])

    # 恢复角度输出格式
    restored_angle = restore_angle_outputs(angle_outputs)
    
    outputs = []
    for i, x in enumerate(concat_outputs):
        # 确定stride和index
        if x.shape[2] == 20: 
            stride = 32
            index = 20 * 4 * 20 * 4 + 20 * 2 * 20 * 2
        elif x.shape[2] == 40:  
            stride = 16
            index = 20 * 4 * 20 * 4
        elif x.shape[2] == 80:  
            stride = 8
            index = 0
        else:
            stride = input_size // x.shape[3]
            index = 0
            print(f"Unknown feature map size: {x.shape}, using stride: {stride}")
        
        # 处理特征图
        feature = x.reshape(1, 79, -1)
        output = process(feature, x.shape[3], x.shape[2], stride, restored_angle, index)
        outputs = outputs + output

    return outputs

def restore_angle_outputs(angle_outputs):
    """
    将角度输出恢复成模型定义中的格式
    """
    bs = angle_outputs[0].shape[0]  # batch size = 1
    ne = angle_outputs[0].shape[1]  # num_angle = 1
    
    # 模拟模型中的reshape操作
    angle_reshaped_list = []
    for i in range(len(angle_outputs)):
        H, W = angle_outputs[i].shape[2], angle_outputs[i].shape[3]
        angle_reshaped = angle_outputs[i].reshape(bs, ne, -1)
        angle_reshaped_list.append(angle_reshaped)
    
    # 模拟模型中的cat操作
    angle_restored = np.concatenate(angle_reshaped_list, axis=2)
    
    # 应用sigmoid
    angle_sigmoid = 1 / (1 + np.exp(-angle_restored))

    return angle_sigmoid

def process(out, model_w, model_h, stride, angle_feature, index, scale_w=1, scale_h=1):
    class_num = len(CLASSES)
    angle_feature = angle_feature.reshape(-1)

    xywh = out[:, :64, :]
    conf = sigmoid(out[:, 64:, :])
    out_boxes = []
    conf = conf.reshape(-1)
    
    for ik in range(model_h * model_w * class_num):
        if conf[ik] > objectThresh:
            w = ik % model_w
            h = (ik % (model_w * model_h)) // model_w
            c = ik // (model_w * model_h)
            xywh_ = xywh[0, :, (h * model_w) + w]
            xywh_ = xywh_.reshape(1, 4, 16, 1)
            data = np.array([i for i in range(16)]).reshape(1, 1, 16, 1)
            xywh_ = softmax(xywh_, 2)
            xywh_ = np.multiply(data, xywh_)
            xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)
            xywh_add = xywh_[:2] + xywh_[2:]
            xywh_sub = (xywh_[2:] - xywh_[:2]) / 2
            angle_feature_ = (angle_feature[index + (h * model_w) + w] - 0.25) * 3.1415927410125732
            angle_feature_cos = math.cos(angle_feature_)
            angle_feature_sin = math.sin(angle_feature_)
            xy_mul1 = xywh_sub[0] * angle_feature_cos
            xy_mul2 = xywh_sub[1] * angle_feature_sin
            xy_mul3 = xywh_sub[0] * angle_feature_sin
            xy_mul4 = xywh_sub[1] * angle_feature_cos
            xywh_1 = np.array([(xy_mul1 - xy_mul2) + w + 0.5, (xy_mul3 + xy_mul4) + h + 0.5, xywh_add[0], xywh_add[1]])
            xywh_ = xywh_1 * stride
            xmin = (xywh_[0] - xywh_[2] / 2)
            ymin = (xywh_[1] - xywh_[3] / 2)
            xmax = (xywh_[0] + xywh_[2] / 2)
            ymax = (xywh_[1] + xywh_[3] / 2)
            box = DetectBox(c, conf[ik], xmin, ymin, xmax, ymax, angle_feature_)
            out_boxes.append(box)
    return out_boxes

def save_obb_result(image_path, predbox, save_dir, offset_x=0, offset_y=0, aspect_ratio=1.0):
    """
    保存OBB检测结果
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    image = cv.imread(image_path)

    # 绘制旋转矩形
    for index in range(len(predbox)):
        xmin = int((predbox[index].xmin - offset_x) / aspect_ratio)
        ymin = int((predbox[index].ymin - offset_y) / aspect_ratio)
        xmax = int((predbox[index].xmax - offset_x) / aspect_ratio)
        ymax = int((predbox[index].ymax - offset_y) / aspect_ratio)
        classId = predbox[index].classId
        score = predbox[index].score
        angle = predbox[index].angle
        
        points = rotate_rectangle(xmin, ymin, xmax, ymax, angle)
        cv.polylines(image, [np.asarray(points, dtype=int)], True, (0, 255, 0), 2)

        ptext = (xmin, ymin)
        title = CLASSES[classId] + " %.2f" % score
        cv.putText(image, title, ptext, cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)

    # 保存图像和检测结果
    file_name = os.path.split(image_path)[-1]
    cv.imwrite(os.path.join(save_dir, file_name), image)
    
    # 保存检测框信息到txt文件
    txt_file = os.path.join(save_dir, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))
    with open(txt_file, 'w') as f:
        for box in predbox:
            xmin = int((box.xmin - offset_x) / aspect_ratio)
            ymin = int((box.ymin - offset_y) / aspect_ratio)
            xmax = int((box.xmax - offset_x) / aspect_ratio)
            ymax = int((box.ymax - offset_y) / aspect_ratio)
            f.write(f"{CLASSES[box.classId]} {box.score:.4f} {xmin} {ymin} {xmax} {ymax} {box.angle:.4f}\n")

def letterbox_resize(image, size):
    if isinstance(image, str):
        image = cv.imread(image)

    target_width, target_height = size
    image_height, image_width, _ = image.shape

    # Calculate the adjusted image size
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    # Create a new canvas and fill it
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2

    return aspect_ratio, offset_x, offset_y

parse = argparse.ArgumentParser(description="RUN YOLOv12-OBB WITH VSX")
parse.add_argument(
    "--file_path",
    type=str,
    default="/source/data/test/",
    help="img or dir path",
)
parse.add_argument("--model_prefix_path", type=str, default="./deploy_weights/ultralytics_yolov12n_obb_fp16/mod", help="model info")
parse.add_argument(
    "--vdsp_params_info",
    type=str,
    default="../build_in/vdsp_params/official-yolov12n-vdsp_params.json", 
    help="vdsp op info",
)
parse.add_argument("--device_id", type=int, default=0, help="device id")
parse.add_argument("--batch", type=int, default=1, help="batch size")
parse.add_argument("--save_dir", type=str, default="./output_vsx/", help="save_dir")
args = parse.parse_args()

class Detector:
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
        assert vsx.set_device(self.device_id) == 0
        
        # 构建model
        model_path = model_prefix_path
        self.model = vsx.Model(model_path, batch_size)
        
        # 输入预处理op
        self.fusion_op = vsx.Operator.load_ops_from_json_file(vdsp_params_info)[0]

        # 构建graph
        self.graph = vsx.Graph()
        self.model_op = vsx.ModelOperator(self.model)
        self.graph.add_operators(self.fusion_op, self.model_op)

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
        
        # 异步处理
        self.consumer = Thread(target=self.async_receive_infer)
        self.consumer.start()

    def async_receive_infer(self):
        while True:
            try:
                result = None
                if len(self.model_output_op_name) > 0:
                    result = self.infer_stream.get_operator_output(self.model_output_op_name)
                else:
                    result = self.infer_stream.get_operator_output(self.model_op)
                
                if result is not None:
                    self.current_id += 1
                    input_id, height, width = self.input_dict[self.current_id]
                    
                    # 获取所有输出
                    outputs_model = []
                    for out in result[0]:
                        outputs_model.append(vsx.as_numpy(out)[0].astype(np.float32))
                    
                    # 使用YOLOv12-OBB后处理
                    detected_boxes = process_yolov12_obb_output(outputs_model, self.model_size)
                    
                    # NMS处理
                    predbox = NMS(detected_boxes)
                    
                    self.result_dict[input_id] = predbox
                    self.event_dict[input_id].set()
                    
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break

    def finish(self):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()

    def _run(self, image: Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv.imread(image, cv.IMREAD_COLOR)
            if cv_image.shape[0] % 2 != 0 or cv_image.shape[1] % 2 != 0:
                cv_image = cv.resize(cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            image = np.stack(cv.split(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)))
        
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        nv12_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)
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

    def run_sync(self, image: Union[str, np.ndarray]):
        """同步推理"""
        if isinstance(image, str):
            cv_image = cv.imread(image, cv.IMREAD_COLOR)
            if cv_image.shape[0] % 2 != 0 or cv_image.shape[1] % 2 != 0:
                cv_image = cv.resize(cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            image = np.stack(cv.split(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)))
        
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
       
        nv12_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)
        yuv_nv12 = nv12_image
        
        output = self.infer_stream.run_sync([yuv_nv12])
        
        # 获取所有输出
        outputs_model = []
        for out in output[0]:
            outputs_model.append(vsx.as_numpy(out)[0].astype(np.float32))
        
        # YOLOv12-OBB后处理
        detected_boxes = process_yolov12_obb_output(outputs_model, self.model_size)
        
        # NMS处理
        predbox = NMS(detected_boxes)

        
        return predbox

if __name__ == '__main__':
    detector = Detector(
        model_prefix_path=args.model_prefix_path,
        vdsp_params_info=args.vdsp_params_info,
        device_id=args.device_id,
        batch_size=args.batch,
        balance_mode=0,
        is_async_infer=False,
        model_output_op_name="", 
    )
    
    if os.path.isfile(args.file_path):
        aspect_ratio, offset_x, offset_y = letterbox_resize(
            args.file_path, (detector.model_size, detector.model_size)
        )
        result = detector.run_sync(args.file_path)
        print(f"{args.file_path} => Detected {len(result)} objects")
        save_obb_result(image, result, args.save_dir, offset_x, offset_y, aspect_ratio)
    else:
        # 批量处理
        images = glob.glob(os.path.join(args.file_path, "*"))
        time_begin = time.time()
        
        for image in images:
            aspect_ratio, offset_x, offset_y = letterbox_resize(
                image, (detector.model_size, detector.model_size)
            )
            result = detector.run_sync(image)
            print(f"{image} => Detected {len(result)} objects")
            save_obb_result(image, result, args.save_dir, offset_x, offset_y, aspect_ratio)
        
        time_end = time.time()
        print(f"\n{len(images)} images in {time_end - time_begin} seconds, ({len(images) / (time_end - time_begin)} images/second)\n")
    
    detector.finish()
    print("test over")
