# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
from queue import Queue
from threading import Thread, Event
from typing import Dict, Generator, Iterable, List, Union
from PIL import Image
import argparse

import vaststreamx as vsx
import glob
import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../../..')
from source_code.util import (calc_nme, compute_fr_and_auc, get_label, concat_output, get_config, pad_crop)

class VSXInference:
    def __init__(self,
        model_prefix_path: Union[str, Dict[str, str]],
        vdsp_params_info: str,
        device_id: int = 0,
        batch_size: int = 1,
        is_async_infer: bool = False,
        model_output_op_name: str = "", ) -> None:


        if isinstance(vdsp_params_info, str):
            with open(vdsp_params_info) as f:
                vdsp_params_info_dict = json.load(f)
        
        self.model_size = vdsp_params_info_dict["OpConfig"]["OimageWidth"]
        self.device_id = device_id
        self.model_output_op_name = model_output_op_name
        self.preprocess_name = "preprocess_res"
        self.input_id = 0

        self.attr = vsx.AttrKey
        # self.device = vsx.set_device(self.device_id)
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
                    pre_process_tensor = self.infer_stream.get_operator_output(self.preprocess_name)
                    # 输出顺序和输入一致 
                    self.current_id += 1
                    input_id,height, width = self.input_dict[self.current_id]
                    model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in result[0]] ]
                    self.result_dict[input_id] = []
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
        output_data = stream_output_list[0]
        self.result_dict[input_id] = output_data

    def _run(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        input_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)

        input_id = self.input_id
        self.input_dict[input_id] = (input_id, height, width)
        self.event_dict[input_id] = Event()


        self.infer_stream.run_async([input_image])
        
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
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3, print(image.shape)
        
        input_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)

        output = self.infer_stream.run_sync([input_image])
        model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in output[0]] ]
        output_data = model_output_list[0]

        return output_data


parser = argparse.ArgumentParser(description="RUN Det WITH VSX")
parser.add_argument(
    "--file_path",
    type=str,
    default= "/path/to/face_alignment/wflw/WFLW ",
    help="img or dir  path",
)
parser.add_argument("--config_path", type = str, default = "../source_code/base_config.yaml", help = "config file")
parser.add_argument("--model_weight_path", type = str, default = "deploy_weights/official_hih_fp16/", help = "model info")
parser.add_argument("--model_name", type = str, default = "mod", help = "model info")
parser.add_argument(
    "--vdsp_params_info",
    type=str,
    default="../vacc_code/vdsp_params/official-hih_wflw_4stack-vdsp_params.json", 
    help="vdsp op info",
)
parser.add_argument("--device_id", type=int, default=0, help="device id")
parser.add_argument("--batch", type=int, default=1, help="bacth size")
parser.add_argument("--save_dir", type=str, default="./output/", help="save_dir")
args = parser.parse_args()


if __name__ == '__main__':
    config = get_config(args.config_path)

    vsx_infer = VSXInference(model_prefix_path=os.path.join(args.model_weight_path, args.model_name),
                        vdsp_params_info=args.vdsp_params_info,
                        device_id=args.device_id,
                        batch_size=args.batch,
                        is_async_infer = False,
                        model_output_op_name = "", 
                    )

    IONs = None
    file_list = glob.glob(os.path.join(args.file_path, "test")+"/*.jpg")
    assert len(file_list)>0, f"FileNotFoundError: {args.file_path}" 
    label_dict = get_label(os.path.join(args.file_path, "test.txt"), ret_dict=True)

    # images = glob.glob(os.path.join(args.file_path, "*"))
    if config.pad_crop:
        for file in tqdm(file_list):
            gt_landmarks = label_dict[os.path.basename(file)]
            cv_image = cv2.imread(file, cv2.IMREAD_COLOR)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            new_image, target = pad_crop(cv_image, gt_landmarks, cv2_flag=True)
            if isinstance(new_image, Image.Image):
                new_image = np.ascontiguousarray(new_image)
                new_image = new_image.astype(np.float32)
            new_image = new_image.transpose((2,0,1))
            result = vsx_infer.run_sync(new_image)

            # image = Image.open(file)
            # if image.mode != 'RGB':
            #     image = image.convert('RGB')
            # if image.size != config.model_shape[2:]:
            #     image = image.resize(config.model_shape[2:])
            # new_image, target = pad_crop(image, gt_landmarks)
            # new_image = np.ascontiguousarray(new_image)
            # new_image = new_image.astype(np.float32)
            # new_image = new_image.transpose((2, 0, 1))
            # result = vsx_infer.run_sync(new_image)
            
            pred0 = np.expand_dims(result[0], axis=0)
            pred1 = np.expand_dims(result[1], axis=0)
            pred_heatmap, pred_offset = concat_output((pred0, pred1))
            sum_ion, ion = calc_nme(config, gt_landmarks, pred_heatmap, pred_offset)
            IONs = np.concatenate((IONs,ion),0) if IONs is not None else ion

    else:
        results = vsx_infer.run_batch(file_list)
        for (image, result) in tzip(file_list, results):
            gt_landmarks = label_dict[os.path.basename(image)]
            pred0 = np.expand_dims(result[0], axis=0)
            pred1 = np.expand_dims(result[1], axis=0)
            pred_heatmap, pred_offset = concat_output((pred0, pred1))
            sum_ion, ion = calc_nme(config, gt_landmarks, pred_heatmap, pred_offset)
            IONs = np.concatenate((IONs,ion),0) if IONs is not None else ion
    compute_fr_and_auc(IONs)   

    vsx_infer.finish()
    print("test over")
