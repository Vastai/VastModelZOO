from curses import has_key
import vaststreamx as vsx
import numpy as np
import argparse
import glob
from tqdm import tqdm
import os
import cv2 as cv
from typing import Dict, Generator, Iterable, List, Union
import json
import torch
from threading import Thread, Event
import time
from queue import Queue

import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../../..')

def clip_coords(boxes, shape):
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

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

    clip_coords(coords, img0_shape)
    return coords


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.
    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.
    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def nms(dets):
    thresh = 0.45
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def post_process(net_outs, det_scale, img, thresh=0.5, use_kps=True, input_height=640, input_width=640):
    scores_list = []
    bboxes_list = []
    kpss_list = []
    fmc = 3
    feat_stride_fpn = [8, 16, 32]
    num_anchors =2
    max_num =0
    metric='default'
    for idx, stride in enumerate(feat_stride_fpn):
        # If model support batch dim, take first output

        scores = net_outs[idx]
        bbox_preds = net_outs[idx + fmc]
        bbox_preds = bbox_preds * stride
        if use_kps:
            kps_preds = net_outs[idx + fmc * 2] * stride

        height = input_height // stride
        width = input_width // stride
        K = height * width
        key = (height, width, stride)

        #solution-3:
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        #print(anchor_centers.shape)

        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if num_anchors>1:
            anchor_centers = np.stack([anchor_centers]*num_anchors, axis=1).reshape((-1,2))

        pos_inds = np.where(scores>=thresh)[0]
        #bbox_preds = bbox_preds.reshape((-1,4))
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)
        if use_kps:
            kpss = distance2kps(anchor_centers, kps_preds)
            #kpss = kps_preds
            kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

    #return scores_list, bboxes_list, kpss_list
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    #bboxes = bboxes_list
    # bboxes = np.vstack(bboxes_list) / det_scale
    bboxes_list = np.vstack(bboxes_list)
    bboxes_list[:, ::2] = bboxes_list[:, ::2] 
    bboxes_list[:, 1::2] = bboxes_list[:, 1::2] 
    bboxes = bboxes_list

    # 这里要根据letterbox原理把bboxes还原到原图大小
    origin_img = img
    bboxes = scale_coords([input_height, input_width], bboxes, origin_img.shape).round()
    
    if use_kps:
        # kpss = np.vstack(kpss_list) / det_scale
        kpss_list = np.vstack(kpss_list)
        kpss_list[:, ::2] = kpss_list[:, ::2] 
        kpss_list[:, 1::2] = kpss_list[:, 1::2] 
        kpss = kpss_list
    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = nms(pre_det)
    det = pre_det[keep, :]
    if use_kps:
        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]
    else:
        kpss = None

    if max_num > 0 and det.shape[0] > max_num:
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
        img_center = img.shape[0] // 2, img.shape[1] // 2
        offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        if metric=='max':
            values = area
        else:
            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
        bindex = np.argsort(
            values)[::-1]  # some extra weight on the centering
        bindex = bindex[0:max_num]
        det = det[bindex, :]
        if kpss is not None:
            kpss = kpss[bindex, :]
    return det, kpss

parse = argparse.ArgumentParser(description="RUN Det WITH VSX")
parse.add_argument("--file_path",type=str,default= "/path/to/widerface/val/images/",help="img or dir  path",)
parse.add_argument("--model_prefix_path", type=str, default="deploy_weights/official_scrfd_run_stream_int8/mod", help="model info")
parse.add_argument("--vdsp_params_info",type=str,default="insightface-scrfd_500m-vdsp_params.json", help="vdsp op info",)
parse.add_argument("--device_id", type=int, default=0, help="device id")
parse.add_argument("--batch", type=int, default=1, help="bacth size")
parse.add_argument("--save_dir", type=str, default="./output/", help="save_dir")
args = parse.parse_args()

def save_result(image_path, result, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    COLORS = np.random.uniform(0, 255, size=(200, 3))
    origin_img = cv.imread(image_path)
    file_dir = image_path.split('/')[-2]
    save_txt = image_path.split('/')[-1].replace('jpg', 'txt')
    
    if not os.path.exists(os.path.join(save_dir, file_dir)):
        os.makedirs(os.path.join(save_dir, file_dir))
    ff = open(os.path.join(save_dir, file_dir, save_txt), 'w')
    ff.write('{:s}\n'.format('%s/%s' % (file_dir, image_path.split('/')[-1][:-4])))
    ff.write('{:d}\n'.format(len(result)))
    for box in result:
        p1, p2 = (int(box["bbox"][0]), int(box["bbox"][1])), (
            int(box["bbox"][2]) + int(box["bbox"][0]),
            int(box["bbox"][3]) + int(box["bbox"][1]),
        )
        cv.rectangle(origin_img, p1, p2, (255, 255, 0), thickness=1, lineType=cv.LINE_AA)
        text = f"{box['label']}: {round(box['score'] * 100, 2)}%"
        y = int(int(box["bbox"][1])) - 15 if int(int(box["bbox"][1])) - 15 > 15 else int(int(box["bbox"][1])) + 15
        cv.putText(
            origin_img,
            text,
            (int(box["bbox"][0]), y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
        ff.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(box['bbox'][0], box['bbox'][1], box['bbox'][2], box['bbox'][3], box['score']))
    ff.close()
    file_name = os.path.split(image_path)[-1]
    # cv.imwrite(os.path.join(save_dir, file_name), origin_img)

class FaceDetector:
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
        self.balance_mode = {0:vsx.StreamBalanceMode.ONCE, 1:vsx.StreamBalanceMode.RUN}

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
                    input_id,height, width, raw_image_path = self.input_dict[self.current_id]
                    model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in result[0]] ]
                    self.result_dict[input_id] = []
                    self.post_processing(input_id, raw_image_path, height, width, model_output_list)
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break
    
    def finish(self,):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()

    def post_processing(self, input_id, raw_img_path, height, width, stream_output_list):
        img_raw = cv.imread(raw_img_path)
        FROM_PYTORCH = True
        stream_ouput_data = stream_output_list[0]
        shape = [(1, 2, 80, 80), (1, 2, 40, 40), (1, 2, 20, 20), (1, 8, 80, 80), (1, 8, 40, 40), (1, 8, 20, 20), (1, 20, 80, 80), (1, 20, 40, 40), (1, 20, 20, 20)]

        net_outs = []
        for i in range(len(stream_ouput_data)):
            net_outs.append(stream_ouput_data[i].reshape(shape[i]))
        
        for i in range(3):
            net_outs[i] = torch.from_numpy(net_outs[i]).float()
            net_outs[i] = net_outs[i].permute(0, 2, 3, 1).reshape(-1, 1).sigmoid()
        for i in range(3, 6):
            net_outs[i] = torch.from_numpy(net_outs[i])
            net_outs[i] = net_outs[i].permute(0, 2, 3, 1).reshape(-1, 4)
        if len(stream_ouput_data) == 9:
            for i in range(6, 9):
                net_outs[i] = torch.from_numpy(net_outs[i])
                net_outs[i] = net_outs[i].permute(0, 2, 3, 1).reshape(-1, 10)

        for i in range(len(stream_ouput_data)):
            net_outs[i] = net_outs[i].numpy()

        bboxes, kpss = post_process(net_outs, [640/width, 640/height], img_raw, use_kps=len(stream_ouput_data)==9)

        for i, det in enumerate(bboxes):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                box = det[:4]
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]

                self.result_dict[input_id].append(
                    {
                        "category_id": 0,
                        "label": 'face',
                        "bbox": box,
                        "score": float(det[4]),
                    }
                )

    def _run(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv.imread(image, cv.IMREAD_COLOR)
            raw_image_path = image
            if cv_image.shape[0]%2 != 0 or cv_image.shape[1]%2 != 0:
                cv_image = cv.resize(cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            image = np.stack(cv.split(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        nv12_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device)
        yuv_nv12 = nv12_image#vsx.cvtcolor(nv12_image, vsx.ImageFormat.YUV_NV12)
        
        input_id = self.input_id
        self.input_dict[input_id] = (input_id, height, width, raw_image_path)
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
            img_raw_path = image
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

        net_outs=[]
        num_out = len(model_output_list[0])
        for i in range(num_out):
            net_outs.append(np.expand_dims(model_output_list[0][i], axis=0))
        
        for i in range(3):
            net_outs[i] = torch.from_numpy(net_outs[i])
            net_outs[i] = net_outs[i].permute(0, 2, 3, 1).reshape(-1, 1).sigmoid()
        for i in range(3, 6):
            net_outs[i] = torch.from_numpy(net_outs[i])
            net_outs[i] = net_outs[i].permute(0, 2, 3, 1).reshape(-1, 4)
        if num_out == 9:
            for i in range(6, 9):
                net_outs[i] = torch.from_numpy(net_outs[i])
                net_outs[i] = net_outs[i].permute(0, 2, 3, 1).reshape(-1, 10)

        for i in range(num_out):
            net_outs[i] = net_outs[i].numpy()

        det_scale = [640/width, 640/height]
        bboxes, kpss = post_process(net_outs, det_scale, img_raw_path, use_kps=num_out==9)
        
        for i, det in enumerate(bboxes):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                box = det[:4]
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]
                result.append.append(
                    {
                        "category_id": 0,
                        "label": 'face',
                        "bbox": box,
                        "score": float(det[4]),
                    }
                )

        return result


if __name__ == '__main__':
    detector = FaceDetector(model_prefix_path=args.model_prefix_path,
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
        save_result(args.file_path, result, args.save_dir)
    else:
        # Test multiple images
        images = glob.glob(os.path.join(args.file_path, "*/*"))
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

