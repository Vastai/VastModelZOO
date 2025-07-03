# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :          Steven
@Email : xinghe@vastaitech.com
@Time  : 	2025/06/20 16:19:31
'''


import os
import sys
import ast
import argparse
from easydict import EasyDict as edict

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

from uie_vacc import UIEVacc
from model_profiler import ModelProfiler
from easydict import EasyDict as edict
import argparse
import ast


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="./build_model_2layer_work/gemma2b_iter0_2048_fp16",
        help="model prefix of the model suite files (default: %(default)s)",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite (default: %(default)s)",
    )
    parser.add_argument(
        "--vdsp_params",
        default="information_extraction/uie/vacc_code/vdsp_params/hustai-uie_base-vdsp_params.json",
        help="vdsp preprocess parameter file (default: %(default)s)",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="./uie_base_pytorch",
        help="tokenizer path (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--device_ids",
        default="[0]",
        help="device ids to run",
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
        "--iterations",
        default=1024,
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
    model_prefix = args.model_prefix
    vdsp_params = args.vdsp_params
    hw_config = args.hw_config
    tokenizer_path = args.tokenizer_path
    device_ids = ast.literal_eval(args.device_ids)
    batch_size = args.batch_size
    instance = args.instance
    iterations = args.iterations
    queue_size = args.queue_size
    input_host = args.input_host
    percentiles = ast.literal_eval(args.percentiles)

    models = []
    contexts = []

    for i in range(instance):
        device_id = device_ids[i % len(device_ids)]
        model = UIEVacc(
            model_prefix=model_prefix, 
            vdsp_config=vdsp_params, 
            tokenizer_path = tokenizer_path, 
            batch_size=batch_size, 
            device_id=device_id, 
            hw_config=hw_config
        )
        models.append(model)
        if input_host:
            contexts.append("CPU")
        else:
            contexts.append("VACC")

    shape = [models[0].input_shape[0]] * 6
    config = edict(
        {
            "instance": instance,
            "iterations": iterations,
            "batch_size": batch_size,
            "data_type": "int32",
            "device_ids": device_ids,
            "contexts": contexts,
            "input_shape": shape,
            "percentiles": percentiles,
            "queue_size": queue_size,
        }
    )
    profiler = ModelProfiler(config, models)
    print(profiler.profiling())
