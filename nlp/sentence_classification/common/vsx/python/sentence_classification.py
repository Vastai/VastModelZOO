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
import cv2 as cv
from typing import Dict, Generator, Iterable, List, Union
import json
from threading import Thread, Event
import time
from queue import Queue
import copy
from PIL import Image

parse = argparse.ArgumentParser(description="RUN Det WITH VSX")
parse.add_argument(
    "--file_path",
    type=str,
    default="./work/VSX/output/0801.png",  
    help="img or dir  path",
)
parse.add_argument("--model_prefix_path", type=str, default="./bert_base_zh_mcls_128-fp16-none-mutil_input-vacc/mod", help="model info")
parse.add_argument(
    "--vdsp_params_info",
    type=str,
    default="./text_recognition/crnn/build_in/vdsp_params/ppocr-resnet34_vd-vdsp_params.json",
    help="vdsp op info",
)
parse.add_argument(
    "--label_txt", type=str, default="test_img/data/images/imagenet.txt", help="label txt"
)
parse.add_argument("--device_id", type=int, default=0, help="device id")
parse.add_argument("--batch", type=int, default=1, help="bacth size")
parse.add_argument("--save_dir", type=str, default="./output", help="save_dir")
parse.add_argument("--save_result_txt", type=str, default="result.txt", help="save result")
args = parse.parse_args()



def save_result(file_path, result, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    features = np.squeeze(result[0]["features"])

    save_txt = file_path.split('/')[-1].replace('.npz', '.txt')
    fin = open(os.path.join(save_dir, save_txt), "w")
    fin.write('{} {:d}\n'.format(file_path, np.argmax(features)))
    fin.close()
    


class Classify:
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
        self.preprocess_name = "preprocess_res"
        self.input_id = 0
        self.balance_mode = {0:vsx.StreamBalanceMode.ONCE, 1:vsx.StreamBalanceMode.RUN}

        self.attr = vsx.AttrKey
        assert vsx.set_device(self.device_id)==0
        # 构建model，模型三件套目录
        model_path = model_prefix_path
        self.model = vsx.Model(model_path, batch_size)
        # 输入预处理op
        # RGB_PLANAR
        self.embedding_op = vsx.Operator(vsx.OpType.BERT_EMBEDDING_OP)
        # 有以上op时无法载通过vsx.Operator加载vdsp算子
        # self.fusion_op = vsx.Operator.load_ops_from_json_file(vdsp_params_info)[0]
        
        
        # 构建graph
        self.graph = vsx.Graph()
        self.model_op = vsx.ModelOperator(self.model)
        self.graph.add_operators(self.embedding_op, self.model_op)#

        # 构建stream
        self.infer_stream = vsx.Stream(self.graph, self.balance_mode[balance_mode])
        if is_async_infer and len(self.model_output_op_name) > 0:
            self.infer_stream.register_operator_output(self.model_output_op_name, self.model_op)
        else:
            self.infer_stream.register_operator_output(self.model_op)
        
        # # 预处理算子输出
        # n,c,h,w = self.model.input_shape[0]
        # self.infer_stream.register_operator_output(self.preprocess_name, self.fusion_op, [[(c,h,w), vsx.TypeFlag.FLOAT16]])

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
                    #pre_process_tensor = self.infer_stream.get_operator_output(self.preprocess_name)
                    # 输出顺序和输入一致 
                    self.current_id += 1
                    input_id, = self.input_dict[self.current_id]
                    model_output_list = [ [vsx.as_numpy(out).astype(np.float32) for out in result[0]] ]
                    self.result_dict[input_id] = []
                    self.post_processing(self.classes, input_id, 0, 0, model_output_list)
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
        output_data = stream_output_list[0][0]
        
        self.result_dict[input_id].append(
            {
                "features": output_data,
            }
        )

    def _run(self, inputs: List[np.ndarray]):
        vsx_tensors = []
        for input in inputs:
            vsx_tensors.append(vsx.from_numpy(input, self.device_id))
        
        ############## compiler 1.5+, 6个input################################
        for i in range(3):
            vsx_tensors.append(vsx.from_numpy(inputs[2], self.device_id))
        #####################################################################

        input_id = self.input_id
        self.input_dict[input_id] = (input_id,)
        self.event_dict[input_id] = Event()


        self.infer_stream.run_async([vsx_tensors])
        
        self.input_id += 1

        return input_id

    def run(self, inputs: List[np.ndarray]):
        input_id = self._run(inputs)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]

        return result

    def run_batch(self, inputs_list: Iterable[List[np.ndarray]]) -> Generator[str, None, None]:
        queue = Queue(20)
        
        def input_thread():
            for inputs in inputs_list:
                input_id = self._run(inputs)
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
    
    def run_sync(self, inputs: List[np.ndarray]):
        vsx_tensors = []
        for input in inputs:
            vsx_tensors.append(vsx.from_numpy(input, self.device_id))
        
        ############## compiler 1.5+, 6个input################################
        for i in range(3):
            vsx_tensors.append(vsx.from_numpy(inputs[2], self.device_id))
        #####################################################################

        output = self.infer_stream.run_sync([vsx_tensors])
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
    classify = Classify(model_prefix_path=args.model_prefix_path,
                        vdsp_params_info=args.vdsp_params_info,
                        classes=args.label_txt,
                        device_id=args.device_id,
                        batch_size=args.batch,
                        balance_mode = 0,
                        is_async_infer = False,
                        model_output_op_name = "", 
                    )
    if os.path.isfile(args.file_path):
        inputs = [np.ones((128,), dtype=np.int32), 
                np.ones((128,), dtype=np.int32),
                np.ones((128,), dtype=np.int32)]#np.load(args.file_path)
        result = classify.run_sync(inputs)
        # print(f"{args.file_path} => {result}")
        save_result(args.file_path, result, args.save_dir)
    else:
        # Test multiple images
        npz_file = glob.glob(os.path.join(args.file_path,  "*.png"))
        # inputs_list = []
        # for npz in npz_file:
        #     inputs = np.load(npz)
        #     inputs_list.append([inputs['input_0'],inputs['input_1'],inputs['input_2']])
        inputs_list = [[np.ones((128,), dtype=np.int32), 
                        np.ones((128,), dtype=np.int32),
                        np.ones((128,), dtype=np.int32)] for i in range(16)]
        time_begin = time.time()
        results = classify.run_batch(inputs_list)
        for (npz, result) in zip(npz_file, results):
            # print(f"{image} => {result}")
            save_result(npz, result, args.save_dir)
        time_end = time.time()

        print(
            f"\n{len(npz_file)} images in {time_end - time_begin} seconds, ({len(npz_file) / (time_end - time_begin)} images/second)\n"
        )
    classify.finish()
    print("test over")
    
