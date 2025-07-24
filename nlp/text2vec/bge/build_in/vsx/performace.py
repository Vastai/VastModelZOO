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

import vaststreamx as vsx
from base import EmbeddingX
from model_profiler import ModelProfiler
from easydict import EasyDict as edict
import argparse
import ast


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/bert_base_en_qa-int8-max-1_384_1_384_1_384-vacc/bert_base_en",
        help="model prefix of the model suite files",
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
        default=1,
        type=int,
        help="cache input data into host memory",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    model_prefix = args.model_prefix
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
        model = EmbeddingX(model_prefix_path=model_prefix, batch_size=batch_size, device_id=device_id)
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
    # print(profiler.profiling())
    result = profiler.profiling()
    print("get result")
    print(result)

    for model in models:
        model.finish()




"""
Usage:
python3 performace.py \
--model_prefix ./MZ-695/bge-m3-512-hunjin-newai/mod \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 100 \
--percentiles "[50, 90, 95, 99]" \
--input_host 1 \
--queue_size 1 


Result:

- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 19.16
  latency (us):
    avg latency: 154957
    min latency: 59912
    max latency: 162722
    p50 latency: 156322
    p90 latency: 156539
    p95 latency: 156580
    p99 latency: 156755

"""