# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys

from detection3d import Detection3D
from model_profiler import ModelProfiler
from easydict import EasyDict as edict
import argparse
import ast


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefixs",
        default="[/path/to/pointpillar_int8/mod]",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_configs",
        default="[]",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--elf_file",
        default="/opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op",
        help="elf file path",
    )
    parser.add_argument(
        "--max_voxel_num",
        default="[16000]",
        help="model max voxel number",
    )
    parser.add_argument(
        "--max_points_num",
        default=120000,
        type=int,
        help="max_points_num to run",
    )
    parser.add_argument(
        "--voxel_size",
        default="[0.16, 0.16, 4]",
        help="voxel size",
    )
    parser.add_argument(
        "--coors_range",
        default="[0, -39.68, -3, 69.12, 39.68, 1]",
        help="coors range",
    )
    parser.add_argument(
        "-d",
        "--device_ids",
        default="[0]",
        help="device ids to run",
    )
    parser.add_argument(
        "--shuffle_enabled",
        default=0,
        type=int,
        help="shuffle enabled",
    )
    parser.add_argument(
        "--normalize_enabled",
        default=0,
        type=int,
        help="normalize enabled",
    )
    parser.add_argument(
        "--feat_size",
        default="[864,496,480,480]",
        help="set model feature sizes,[max_feature_width,max_feature_height,actual_feature_width,actual_feature_height]",
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
    model_prefixs = args.model_prefixs.strip("[").strip("]").split(",")
    hw_configs = args.hw_configs.strip("[").strip("]").split(",")
    device_ids = ast.literal_eval(args.device_ids)
    batch_size = args.batch_size
    instance = args.instance
    iterations = args.iterations
    queue_size = args.queue_size
    input_host = args.input_host
    percentiles = ast.literal_eval(args.percentiles)
    max_voxel_nums = ast.literal_eval(args.max_voxel_num)
    max_points_num = args.max_points_num

    voxel_size = ast.literal_eval(args.voxel_size)
    coors_range = ast.literal_eval(args.coors_range)
    feat_size = ast.literal_eval(args.feat_size)

    assert len(model_prefixs) == len(max_voxel_nums)
    assert len(feat_size) == 4

    models = []
    contexts = []

    modle_configs = [
        edict(
            {
                "max_voxel_num": max_voxel_nums[i],
                "model_prefix": model_prefixs[i],
                "hw_config": hw_configs[i] if len(hw_configs) > i else "",
            }
        )
        for i in range(len(model_prefixs))
    ]

    for i in range(instance):
        device_id = device_ids[i % len(device_ids)]
        model = Detection3D(
            model_configs=modle_configs,
            elf_file=args.elf_file,
            voxel_sizes=voxel_size,
            coors_range=coors_range,
            device_id=device_id,
            max_points_num=max_points_num,
            shuffle_enabled=args.shuffle_enabled,
            normalize_enabled=args.normalize_enabled,
            max_feature_width=feat_size[0],
            max_feature_height=feat_size[1],
            actual_feature_width=feat_size[2],
            actual_feature_height=feat_size[3],
        )
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
            "data_type": "float16",
            "device_ids": device_ids,
            "contexts": contexts,
            "input_shape": shape,
            "percentiles": percentiles,
            "queue_size": queue_size,
        }
    )
    profiler = ModelProfiler(config, models)
    print(profiler.profiling())
