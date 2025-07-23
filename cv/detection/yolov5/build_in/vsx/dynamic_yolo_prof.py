# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :          lance
@Email : lance.wang@vastaitech.com
@Time  : 	2025/04/21 19:43:31
'''

import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "./")
sys.path.append(common_path)

from common.dynamic_detector import DynamicDetector
from common.model_profiler import ModelProfiler
from easydict import EasyDict as edict
import argparse
import ast


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--module_info",
        default="/path/to/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--vdsp_params",
        default="./data/configs/yolo_div255_bgr888.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--max_input_shape",
        default="[1,3,640,640]",
        help="model max input shape",
    )
    parser.add_argument(
        "-d",
        "--device_ids",
        default="[0]",
        help="device id to run",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=1,
        type=int,
        help="profiling batch size of the model",
    )
    parser.add_argument(
        "-i",
        "--instance",
        default=1,
        type=int,
        help="instance number for each device",
    )
    parser.add_argument(
        "-s",
        "--shape",
        help="data input shape",
    )
    parser.add_argument(
        "--model_input_shape",
        default="[1,3,640,640]",
        help="model input shape",
    )
    parser.add_argument(
        "--iterations",
        default=10240,
        type=int,
        help="iterations count for one profiling",
    )
    parser.add_argument(
        "--queue_size",
        default=1,
        type=int,
        help="aync wait queue size",
    )
    parser.add_argument(
        "--percentiles",
        default="[50, 90, 95, 99]",
        help="percentiles of latency",
    )
    parser.add_argument(
        "--input_host",
        default=0,
        type=int,
        help="cache input data into host memory",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    model_prefix = args.module_info
    vdsp_params = args.vdsp_params
    device_ids = ast.literal_eval(args.device_ids)
    batch_size = args.batch_size
    instance = args.instance
    iterations = args.iterations
    queue_size = args.queue_size
    input_host = args.input_host
    percentiles = ast.literal_eval(args.percentiles)
    max_input_shape = ast.literal_eval(args.max_input_shape)
    model_input_shape = ast.literal_eval(args.model_input_shape)

    models = []
    contexts = []

    for i in range(instance):
        device_id = device_ids[i % len(device_ids)]
        model = DynamicDetector(
            args.module_info,
            args.vdsp_params,
            [max_input_shape],
            batch_size,
            device_id,
        )
        model_input_shapes = [model_input_shape] * batch_size
        model.set_model_input_shape(model_input_shapes)
        models.append(model)
        if input_host:
            contexts.append("CPU")
        else:
            contexts.append("VACC")

    if args.shape:
        shape = ast.literal_eval(args.shape)
    else:
        shape = models[0].input_shape[0]
    config = edict(
        {
            "instance": instance,
            "iterations": iterations,
            "batch_size": batch_size,
            "data_type": "uint8",
            "device_ids": device_ids,
            "contexts": contexts,
            "input_shape": shape,
            "percentiles": percentiles,
            "queue_size": queue_size,
        }
    )
    profiler = ModelProfiler(config, models)
    print(profiler.profiling())
