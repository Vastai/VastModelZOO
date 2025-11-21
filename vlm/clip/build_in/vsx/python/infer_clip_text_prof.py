# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys
import queue
from easydict import EasyDict as edict
import argparse
import ast
from typing import Union, List
import vaststreamx as vsx
import numpy as np
import time
import copy
import threading
import clip
from threading import Thread, Event
import concurrent.futures
from clip_model import NormalType, NormalizeOp, cv_rgb_image_to_vastai

class ProfilerResult:
    def __init__(self) -> None:
        self.throughput = 0
        self.latency_avg = 0
        self.latency_max = 0
        self.latency_min = 0
        self.latency_pecents = []
        self.config = None

    def __str__(self) -> str:
        repr_str = "\n- "
        if self.config:
            repr_str += f"""number of instances: {self.config.instance}
  devices: {self.config.device_ids}
  queue size: {self.config.queue_size}
  batch size: {self.config.batch_size}
"""
        repr_str += f"""  throughput (qps): {self.throughput:.2f}
  latency (us):
    avg latency: {self.latency_avg}
    min latency: {self.latency_min}
    max latency: {self.latency_max}
"""
        if self.config:
            for i, pecent in enumerate(self.config.percentiles):
                repr_str += f"    p{pecent} latency: {self.latency_pecents[i]}\n"
        return repr_str


class ClipText():
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        output_type=vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE,
    ):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.preproc_ops_ = vsx.Operator.load_ops_from_json_file(vdsp_config)
        assert len(self.preproc_ops_) > 0, "load_ops_from_json_file failed"
        self.graph_ = vsx.Graph(output_type)
        self.graph_.add_operators(*self.preproc_ops_, self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()
        
    @property
    def batch_size(self):
        return self.model_.batch_size

    @property
    def max_batch_size(self):
        return self.model_.max_batch_size

    @property
    def input_count(self):
        return self.model_.input_count

    @property
    def output_count(self):
        return self.model_.output_count

    @property
    def input_shape(self):
        return self.model_.input_shape

    @property
    def output_shape(self):
        return self.model_.output_shape

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        tokens = self.make_tokens("test string")
        if ctx == "CPU":
            return [tokens] * batch_size
        else:
            return [
                [vsx.from_numpy(token, self.device_id_) for token in tokens]
            ] * batch_size

    def make_tokens(self, text):
        assert isinstance(text, str), f"input type must be str"
        token = clip.tokenize(text)[0]
        token_padding = np.pad(token.numpy(), pad_width=(0, 3)).astype(np.int32)
        # make mask
        index = np.argmax(token_padding)
        token_mask = copy.deepcopy(token_padding)
        token_mask[: index + 1] = 1
        # make input
        zero_arr = np.zeros(token_padding.shape, dtype=np.int32)
        tokens = []
        tokens.append(token_padding)
        tokens.append(zero_arr)
        tokens.append(zero_arr)
        tokens.append(token_mask)
        tokens.append(zero_arr)
        tokens.append(zero_arr)

        return tokens

    def process(
        self,
        input: Union[
            List[List[vsx.Tensor]],
            List[List[np.ndarray]],
            List[vsx.Tensor],
            List[np.ndarray],
            List[str],
            str,
        ],
    ):
        if isinstance(input, list):
            if isinstance(input[0], list):
                if isinstance(input[0][0], np.ndarray):
                    return self.process(
                        [
                            [
                                vsx.from_numpy(
                                    np.array(x, dtype=np.int32), self.device_id_
                                )
                                for x in one
                            ]
                            for one in input
                        ]
                    )
                else:
                    return self.process_impl(input)
            elif isinstance(input[0], str):
                return self.process([self.make_tokens(x) for x in input])
            elif isinstance(input[0], np.ndarray):
                tensors = [
                    vsx.from_numpy(np.array(x, dtype=np.int32), self.device_id_)
                    for x in input
                ]
                return self.process_impl([tensors])[0]
            else:
                return self.process_impl([input])[0]
        else:
            tokens = self.make_tokens(input)
            return self.process(tokens)

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [vsx.as_numpy(out[0]).astype(np.float32) for out in outputs]

class ModelProfiler:
    def __init__(self, config, models) -> None:
        self.config_ = config
        self.models_ = models
        self.iters_left_ = config.iterations
        self.merge_lock = threading.Lock()
        self.throughput_ = 0
        self.latency_begin_ = []
        self.latency_end_ = []
        if config.iterations > 0:
            self.long_time_test_ = False
        else:
            self.long_time_test_ = True

    def profiling(self):
        threads = []
        for i in range(len(self.models_)):
            device_id = self.config_.device_ids[i % len(self.config_.device_ids)]
            # print(f"thread id {i} use device id {device_id}")
            if self.config_.queue_size == 0:
                thread_inst = threading.Thread(
                    target=self.drive_on_latancy_mode, args=(i, device_id)
                )
            else:
                thread_inst = threading.Thread(
                    target=self.drive_one_instance, args=(i, device_id)
                )
            thread_inst.start()
            threads.append(thread_inst)
        for thread_inst in threads:
            thread_inst.join()
        latency_us = (
            np.array(self.latency_end_) - np.array(self.latency_begin_)
        ) * 1000000
        result = ProfilerResult()
        result.latency_pecents = (
            np.percentile(latency_us, self.config_.percentiles + [0, 100])
            .astype("int")
            .tolist()
        )
        result.latency_max = result.latency_pecents.pop()
        result.latency_min = result.latency_pecents.pop()
        result.latency_avg = int(np.mean(latency_us))
        result.throughput = self.throughput_
        result.config = self.config_
        return result

    def process_async(self, model, input):
        vsx.set_device(model.device_id_)
        return model.process(input)

    def drive_one_instance(self, idx, device_id):
        vsx.set_device(device_id)
        infer_data = self.models_[idx].get_test_data(
            self.config_.data_type,
            self.config_.input_shape,
            self.config_.batch_size,
            self.config_.contexts[idx],
        )
        queue_futs = queue.Queue(self.config_.queue_size)
        ticks = []
        tocks = []
        context = edict(stopped=False, left=0)

        def cunsume_thread_func(context, queue_futs, tocks):
            while not context.stopped or context.left > 0:
                try:
                    fut = queue_futs.get(timeout=0.01)
                    fut.result()
                    tock = time.time()
                    if self.long_time_test_ is not True:
                        tocks.append(tock)
                    context.left -= 1
                except queue.Empty:
                    pass

        cunsume_thread = threading.Thread(
            target=cunsume_thread_func, args=(context, queue_futs, tocks)
        )
        cunsume_thread.start()
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while self.iters_left_ >= 0 or self.long_time_test_:
                tick = time.time()
                # fut = executor.submit(self.models_[idx].process, infer_data)
                fut = executor.submit(self.process_async, self.models_[idx], infer_data)
                queue_futs.put(fut)
                self.iters_left_ -= 1
                context.left += 1
                if self.long_time_test_ is not True:
                    ticks.append(tick)
        context.stopped = True
        cunsume_thread.join()
        end = time.time()
        self.merge_lock.acquire()
        time_used = (end - start) * 1000000
        self.throughput_ += len(ticks) * self.config_.batch_size * 1000000.0 / time_used
        assert len(ticks) == len(tocks)
        self.latency_begin_ += ticks
        self.latency_end_ += tocks
        self.merge_lock.release()

    def drive_on_latancy_mode(self, idx, device_id):
        vsx.set_device(device_id)
        infer_data = self.models_[idx].get_test_data(
            self.config_.data_type,
            self.config_.input_shape,
            self.config_.batch_size,
            self.config_.contexts[idx],
        )
        ticks = []
        tocks = []

        start = time.time()
        while self.iters_left_ > 0:
            tick = time.time()
            self.models_[idx].process(infer_data)
            tock = time.time()
            self.iters_left_ -= 1
            tocks.append(tock)
            ticks.append(tick)
        end = time.time()

        self.merge_lock.acquire()
        time_used = (end - start) * 1000000
        self.throughput_ += len(ticks) * self.config_.batch_size * 1000000.0 / time_used
        assert len(ticks) == len(tocks)
        self.latency_begin_ += ticks
        self.latency_end_ += tocks
        self.merge_lock.release()

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        help="hw-config file of the model suite",
        default="",
    )
    parser.add_argument(
        "--vdsp_params",
        default="./data/configs/clip_txt_vdsp.json",
        help="vdsp preprocess parameter file",
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
        model = ClipText(model_prefix, vdsp_params, batch_size, device_id, hw_config)
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
