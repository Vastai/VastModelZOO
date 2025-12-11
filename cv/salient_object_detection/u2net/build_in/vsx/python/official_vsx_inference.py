# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import cv2
import time
import json
import glob
import torch
import argparse
import threading
import numpy as np
from PIL import Image
from tqdm.contrib import tzip
from queue import Queue
from threading import Thread, Event
from typing import Dict, Generator, Iterable, List, Union

import vaststreamx as vsx


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
        output_data = stream_output_list[0][0]
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
        assert c == 3
        
        input_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)

        output = self.infer_stream.run_sync([input_image])
        model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in output[0]] ]
        output_data = model_output_list[0][0]

        return output_data


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def set_config():
    parse = argparse.ArgumentParser(description="RUN WITH VSX")
    parse.add_argument(
        "--image_dir",
        type=str,
        default="/path/to/ECSSD/image",
        help="image_dir img dir",
    )
    parse.add_argument(
        "--model_prefix_path",
        type=str,
        default="deploy_weights/official_u2net_int8/mod",
        help="model info")
    parse.add_argument(
        "--vdsp_params_info",
        type=str,
        default="../build_in/vdsp_params/official-u2net-vdsp_params.json",
        help="vdsp op info",
    )
    parse.add_argument(
        "--onnx_file",
        type=str,
        default="u2net-3_320_320.onnx",
        help="onnx weight file for onnxruntime",
    )
    parse.add_argument("--onnx_infer", action='store_true', default=False, help="whether to use onnxruntime inference")
    parse.add_argument("--device_id", type=int, default=0, help="device id")
    parse.add_argument("--batch", type=int, default=1, help="bacth size")
    parse.add_argument("--save_dir", type=str, default="./vsx_results", help="save result")
    args = parse.parse_args()

    return args


if __name__ == '__main__':
    args = set_config()

    # Test multiple images
    image_files = glob.glob(os.path.join(args.image_dir, "*.jpg"))  #  "*.png"
    # sort images
    image_files.sort()

    os.makedirs(args.save_dir, exist_ok=True)

    #############################################################################
    if args.onnx_infer:
        print("onnxruntime infer...")
        import onnxruntime
        session = onnxruntime.InferenceSession(args.onnx_file)
        # import torch
        # device = torch.device('cpu')
        # model  = torch.jit.load("deploy_weights/model_best.torchscript.pt", map_location=device)
        # model = model.to(device)
        # model.eval()
        results = image_files
    else:
        print("vacc infer...")
        vsx_inference = VSXInference(model_prefix_path=args.model_prefix_path,
                            vdsp_params_info=args.vdsp_params_info,
                            device_id=args.device_id,
                            batch_size=args.batch,
                            is_async_infer = False,
                            model_output_op_name = "", 
                        )
        results = vsx_inference.run_batch(image_files)
    #############################################################################

    for (image_path, result) in tzip(image_files, results):
        if args.onnx_infer:
            #############################################################################
            image_src = cv2.imread(image_path)
            image = cv2.resize(image_src, (320, 320))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image / 255.0
            image = (image - mean) / std
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = np.expand_dims(image, axis=0)

            # for onnxruntime
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            result = session.run([output_name], {input_name: image})
            
            ## for torch.jit
            # with torch.no_grad():
            #     heatmap = model(torch.from_numpy(image))
            # result = np.squeeze(heatmap.detach().numpy())
            #############################################################################

        # post process
        heatmap = np.expand_dims(result, 0)
        pred = torch.from_numpy(heatmap)
        # pred = 1.0 - pred[:,0,:,:]
        pred = normPRED(pred)
        mask = pred.squeeze()
        mask = mask.data.numpy()
        mask = Image.fromarray(mask*255).convert('RGB')
        mask.save(os.path.join(args.save_dir, os.path.basename(image_path)))
        
    if not args.onnx_infer: 
        vsx_inference.finish()
    
"""
ECSSD dataset: http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html
eval: https://github.com/lartpang/PySODEvalToolkit

指标：↓越小越好，↑越大越好
| methods   |  mae↓ |  maxfmeasure↑ |  avgfmeasure↑ |  adpfmeasure↑ |  maxprecision↑ |  avgprecision↑ |  maxrecall↑ |  avgrecall↑ |  maxem↑ |  avgem↑ |  adpem↑ |   sm↑ |  wfm↑ |


u2net_1_3_320_320.onnx
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.033 |         0.941 |         0.922 |         0.891 |          0.983 |          0.941 |           1 |       0.915 |   0.958 |   0.947 |   0.925 | 0.928 | 0.909 |

u2net-fp16-none-1_3_320_320-vacc
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.033 |         0.941 |         0.925 |         0.924 |          0.986 |          0.941 |           1 |       0.917 |   0.957 |   0.948 |   0.951 | 0.928 | 0.911 |

# u2net-int8-percentile-1_3_320_320-vacc
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.034 |         0.939 |         0.921 |         0.922 |          0.984 |          0.938 |           1 |       0.915 |   0.956 |   0.946 |    0.95 | 0.926 | 0.907 |

"""

"""
u2netp_1_3_320_320.onnx
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.041 |         0.928 |         0.902 |         0.874 |          0.986 |          0.927 |           1 |       0.897 |   0.948 |    0.93 |   0.918 | 0.917 | 0.884 |

u2netp-fp16-none-1_3_320_320-vacc
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.041 |         0.927 |         0.904 |         0.905 |           0.99 |          0.928 |           1 |       0.897 |   0.949 |   0.931 |   0.934 | 0.917 | 0.885 |

u2netp-int8-percentile-1_3_320_320-vacc
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.149 |         0.747 |         0.587 |         0.736 |          0.899 |          0.853 |           1 |       0.416 |   0.862 |   0.622 |   0.809 | 0.662 | 0.516 |
"""

"""
Supervisely_Person_Dataset

u2net_human_seg_1_3_320_320.onnx
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.006 |         0.987 |         0.968 |         0.906 |              1 |          0.974 |           1 |       0.967 |   0.996 |   0.985 |    0.92 | 0.977 | 0.968 |

u2net_human_seg_1_3_320_320-fp16-none-1_3_320_320-vacc
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.006 |         0.987 |          0.97 |         0.948 |              1 |          0.974 |           1 |       0.969 |   0.997 |   0.987 |   0.979 | 0.977 | 0.969 |

u2net_human_seg_1_3_320_320-int8-percentile-1_3_320_320-vacc
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.006 |         0.985 |         0.968 |         0.943 |              1 |          0.972 |           1 |       0.968 |   0.996 |   0.986 |   0.972 | 0.975 | 0.967 |

"""