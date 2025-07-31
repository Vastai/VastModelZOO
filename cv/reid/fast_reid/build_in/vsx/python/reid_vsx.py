# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import sys
import os

_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../../../source_code/fast-reid/')

from fastreid.config import get_cfg
from collections import OrderedDict
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch

import cv2
import glob
import argparse
import torch
import hashlib
import tvm
import vaststreamx as vsx
import numpy as np
from tvm.contrib import graph_runtime

from typing import Dict, Generator, Iterable, List, Union
import json
from threading import Thread, Event
import time
from queue import Queue

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


class Engine:
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
        self.input_id = 0

        self.attr = vsx.AttrKey
        assert vsx.set_device(self.device_id)==0
        # 构建model，模型三件套目录
        model_path = model_prefix_path
        self.model = vsx.Model(model_path, batch_size)
        # 输入预处理op
        # RGB_PLANAR
        self.fusion_op = vsx.Operator.load_ops_from_json_file(vdsp_params_info)[
            0]

        # 构建graph
        self.graph = vsx.Graph(False)
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
                        [vsx.as_numpy(out).astype(np.float32) for out in result[0]]]
                    self.result_dict[input_id] = []
                    self.post_processing(input_id, model_output_list)
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break

    def finish(self,):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()

    def post_processing(self, input_id, stream_output_list):
        output_data = stream_output_list[0][0]
        self.result_dict[input_id] = output_data

    def _run(self, image: Union[str, np.ndarray]):
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


def main(args):
    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.DEVICE = "cpu"

    engine = Engine(model_prefix_path=args.model_prefix_path,
                    vdsp_params_info=args.vdsp_params_info,
                    device_id=0,
                    batch_size=1,
                    is_async_infer=False,
                    model_output_op_name="",
                    )

    results = OrderedDict()
    for _, dataset_name in enumerate(cfg.DATASETS.TESTS):
        data_loader, evaluator = DefaultTrainer.build_evaluator(cfg, dataset_name)
        
        for _, inputs in enumerate(data_loader):
            
            outputs = []
            for _, input_data in enumerate(inputs['img_paths']):
                output = engine.run(input_data)
                outputs.append(output)

            outputs = torch.Tensor(outputs).squeeze(1)
            evaluator.process(inputs, outputs)

        results[dataset_name] = evaluator.evaluate()

    if len(results) == 1:
        results = list(results.values())[0]
    print(results)
    
    engine.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RUN Det WITH VSX")
    parser.add_argument("--model_prefix_path", type=str, default="/path/to/reid_model/mod", help="model info")
    parser.add_argument("--vdsp_params_info", type=str, default="../vacc_code/vdsp_params/official-market_bot_R50-vdsp_params.json", help="vdsp op info",)
    parser.add_argument("--device_id", type=int, default=0, help="device id")
    parser.add_argument("--save_dir", type=str,default="./output/", help="save_dir")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--resume", action="store_true", help="whether to attempt to resume from the checkpoint directory",)
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=0, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        0,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

# int8
# OrderedDict([('Rank-1', 92.87410974502563), ('Rank-5', 97.20902442932129), ('Rank-10', 98.15914630889893), ('mAP', 82.10230469703674), ('mINP', 52.90618538856506), ('metric', 87.48821020126343)])
    
# fp16
# OrderedDict([('Rank-1', 91.83491468429565), ('Rank-5', 96.88242077827454), ('Rank-10', 98.04037809371948), ('mAP', 79.13962006568909), ('mINP', 46.81326746940613), ('metric', 85.48727035522461)])
