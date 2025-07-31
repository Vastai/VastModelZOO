# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import threading
from tkinter import Image
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
import copy
from PIL import Image
from tqdm import tqdm
import re

parser = argparse.ArgumentParser(description="RUN Det WITH VSX")
parser.add_argument(
    "--file_path",
    type=str,
    default="./output/A2F3Y4_32_70.jpg",  
    help="img or dir  path",
)
parser.add_argument("--model_prefix_path", type=str, default="crnn", help="model info")
parser.add_argument(
    "--vdsp_params_info",
    type=str,
    default="crnn-vdsp_params_new.json",
    help="vdsp op info",
)

parser.add_argument("--device_id", type=int, default=0, help="device id")
parser.add_argument("--batch", type=int, default=1, help="bacth size")
parser.add_argument("--label", type=str, default="../vacc_code/config/ocr_rec_dict.txt", help="decode label")
parser.add_argument("--output_file", type=str, default="cls_pred.txt", help="save result")
args = parser.parse_args()

def cv_bgr888_to_nv12(bgr888):
    yuv_image = cv2.cvtColor(bgr888, cv2.COLOR_BGR2YUV_I420)
    height, width = bgr888.shape[:2]
    y = yuv_image[:height, :]
    u = yuv_image[height : height + height // 4, :]
    v = yuv_image[height + height // 4 :, :]
    u = np.reshape(u, (height // 2, width // 2))
    v = np.reshape(v, (height // 2, width // 2))
    uv_plane = np.empty((height // 2, width), dtype=np.uint8)
    uv_plane[:, 0::2] = u
    uv_plane[:, 1::2] = v
    yuv_nv12 = np.concatenate((y, uv_plane), axis=0)
    return yuv_nv12

def cv_bgr888_to_vsximage(bgr888, vsx_format, device_id):
    h, w = bgr888.shape[:2]
    if vsx_format == vsx.ImageFormat.BGR_INTERLEAVE:
        res = bgr888
    elif vsx_format == vsx.ImageFormat.BGR_PLANAR:
        res = np.array(bgr888).transpose(2, 0, 1)
    elif vsx_format == vsx.ImageFormat.RGB_INTERLEAVE:
        res = cv2.cvtColor(bgr888, cv2.COLOR_BGR2RGB)
    elif vsx_format == vsx.ImageFormat.RGB_PLANAR:
        res = np.array(cv2.cvtColor(bgr888, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    elif vsx_format == vsx.ImageFormat.YUV_NV12:
        res = cv_bgr888_to_nv12(bgr888=bgr888)
    else:
        assert False, f"Unsupport format:{vsx_format}"
    return vsx.create_image(
        res,
        vsx_format,
        w,
        h,
        device_id,
    )

class Classification:
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
        
        # self.model_size = vdsp_params_info_dict["OpConfig"]["OimageWidth"]
        self.device_id = device_id
        self.model_output_op_name = model_output_op_name
        self.preprocess_name = "preprocess_res"
        self.input_id = 0
        self.balance_mode = {0:vsx.StreamBalanceMode.ONCE, 1:vsx.StreamBalanceMode.RUN}

        self.attr = vsx.AttrKey
        assert vsx.set_device(self.device_id)==0
        # 构建model，模型三件套目录
        model_path = model_prefix_path
        self.model = vsx.Model(model_path, batch_size)
        # 输入预处理op
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
        
        # 预处理算子输出
        n,c,h,w = self.model.input_shape[0]
        self.infer_stream.register_operator_output(self.preprocess_name, self.fusion_op, [[(c,h,w), vsx.TypeFlag.FLOAT16]])
        
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
                    #pre_process_tensor = self.infer_stream.get_operator_output(self.preprocess_name)
                    # 输出顺序和输入一致 
                    self.current_id += 1
                    input_id,height, width = self.input_dict[self.current_id]
                    model_output_list = [ [vsx.as_numpy(out).astype(np.float32) for out in result[0]] ]
                    self.result_dict[input_id] = []
                    #print(f"put featuers: input_id-{input_id},{vsx.as_numpy(result[0][0])[0, 0:5]}")
                    self.post_processing(input_id, height, width, model_output_list)
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break
    
    def finish(self,):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()

    def post_processing(self, input_id, height, width, stream_output_list):
        output_data = stream_output_list[0][0]
        self.result_dict[input_id].append(
            {
                "features": output_data,
            }
        )
    
    def calculate_padding(self, model_width, model_height, image_width, image_height):
        radio = image_width / image_height
        # print(f"image_width:{image_width} image_height:{image_height} radio:{radio}")
        resize_w = 0
        resize_h = model_height
        # n,c,h,w
        if (model_height * radio > model_width) :
            resize_w = model_width
        else:
            resize_w = int(model_height * radio)
        
        right = model_width - resize_w if model_width - resize_w > 0 else 0
        
        # (resize_width , resize_height , top , bottom ,left, right)
        # return  (model_width, model_height, 0, 0, 0, right)
        return (int(resize_w), resize_h, 0, 0, 0, right)

    def _run(self, image:Union[str, np.ndarray]):
        cv_image = cv2.imread(image)
        assert image is not None, f"Failed to read input file: {image}"
        vsx_image = cv_bgr888_to_vsximage(cv_image, vsx.ImageFormat.RGB_PLANAR, args.device_id)
        width = cv_image.shape[1]
        height = cv_image.shape[0]
        ext_op_config = self.calculate_padding(self.model.input_shape[0][3], self.model.input_shape[0][2], width, height)
        # print(ext_op_config)
        # print(vsx.as_numpy(vsx_image))
        
        input_id = self.input_id
        self.input_dict[input_id] = (input_id, height, width)
        self.event_dict[input_id] = Event()


        self.infer_stream.run_async([vsx_image], {
            # "rgb_letterbox_ext" : [(resize_width , resize_height , top , bottom ,left, right)]
            "rgb_letterbox_ext":[ext_op_config] 
        })
        
        self.input_id += 1

        return input_id

    def run(self, image:Union[str, np.ndarray]):
        input_id = self._run(image)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]

        return result

    def run_batch(self, images: Iterable[Union[str, np.ndarray]]) -> Generator[str, None, None]:
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
            #print(f"====>>>>pop(input_id)={input_id}")
            result = self.result_dict.pop(input_id)
            del self.event_dict[input_id]
            yield result
    
    def run_sync(self, image:Union[str, np.ndarray]):
        cv_image = cv2.imread(image)
        assert image is not None, f"Failed to read input file: {image}"
        vsx_image = cv_bgr888_to_vsximage(cv_image, vsx.ImageFormat.RGB_PLANAR, args.device_id)
        width = cv_image.shape[1]
        height = cv_image.shape[0]
        ext_op_config = self.calculate_padding(self.model.input_shape[0][3], self.model.input_shape[0][2], width, height)
        # print(ext_op_config)
        # print(vsx.as_numpy(vsx_image))
        output = self.infer_stream.run_sync([vsx_image], {
                # "rgb_letterbox_ext" : [(resize_width , resize_height , top , bottom ,left, right)]
                "rgb_letterbox_ext":[ext_op_config] 
            })
        model_output_list = [ [vsx.as_numpy(out).astype(np.float32) for out in output[0]] ]
        output_data = model_output_list[0][0]
        
        result = []
        result.append(
            {
            "features": output_data,
            }
        )
        return result
        
        
if __name__ == '__main__':
    text_reco = Classification(model_prefix_path=args.model_prefix_path,
                        vdsp_params_info=args.vdsp_params_info,
                        device_id=args.device_id,
                        batch_size=args.batch,
                        balance_mode = 0,
                        is_async_infer = False,
                        model_output_op_name = "", 
                    )
    if os.path.isfile(args.file_path):
        result = text_reco.run_sync(args.file_path)#run(args.file_path)
        features = np.squeeze(result[0]["features"])
    else:
        # Test multiple images
        images = glob.glob(os.path.join(args.file_path, "*.jpg"))
        images = sorted(images)
        time_begin = time.time()
        results = text_reco.run_batch(images)

        # 写结果数据
        with open(args.output_file, "w") as outfile:
            for (image, result) in zip(images, results):
                #print(f"====>>>>process image: {image}")
                basename, _ = os.path.splitext(os.path.basename(image))
                features = np.squeeze(result[0]["features"])
                print(features)
                preds = features
                if len(preds.shape) == 1:
                    preds = preds.reshape(1, -1)
                label_list = ["0", "180"]
                if label_list is None:
                    label_list = {idx: idx for idx in range(preds.shape[-1])}

                pred_idxs = preds.argmax(axis=1)
                decode_out = [(label_list[idx], preds[i][idx])
                            for i, idx in enumerate(pred_idxs)]
                print(f"img-{image}, result-{decode_out}")
                outfile.write(f"{basename} {decode_out}\n")
                
        outfile.close()

        time_end = time.time()

        print(
            f"\n{len(images)} images in {time_end - time_begin} seconds, ({len(images) / (time_end - time_begin)} images/second)\n"
        )
    
    text_reco.finish()
    print("test over")
    
    
    
