# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import vaststreamx as vsx
import numpy as np
import argparse
import ast
import common.utils as utils
from typing import Union, List
from easydict import EasyDict as edict

import time
import queue
import threading
import concurrent.futures


attr = vsx.AttrKey

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



class ModelBase:
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        do_copy=True,
    ) -> None:
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.preproc_ops_ = vsx.Operator.load_ops_from_json_file(vdsp_config)
        assert len(self.preproc_ops_) > 0, "load_ops_from_json_file failed"
        self.graph_ = vsx.Graph(do_copy)
        self.graph_.add_operators(*self.preproc_ops_, self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

    def get_fusion_op_iimage_format(self):
        for op in self.preproc_ops_:
            if op.op_type >= 100:
                buildin_op = op.cast_to_buildin_operator()
                if "IIMAGE_FORMAT" in list(buildin_op.attributes.keys()):
                    imagetype = buildin_op.get_attribute(attr.IIMAGE_FORMAT)
                    return utils.imagetype_to_vsxformat(imagetype)
                else:
                    return vsx.ImageFormat.YUV_NV12
        assert False, "Can't find fusion op that op_type >= 100"

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



def cv_rgb_image_to_vastai(image_cv, device_id):
    assert len(image_cv.shape) >= 2
    h = image_cv.shape[0]
    w = image_cv.shape[1]
    if len(image_cv.shape) == 3:
        return vsx.create_image(
            image_cv, vsx.ImageFormat.BGR_INTERLEAVE, w, h, device_id
        )
    elif len(image_cv.shape) == 2:
        return vsx.create_image(image_cv, vsx.ImageFormat.GRAY, w, h, device_id)
    else:
        raise Exception("unsupported ndarray shape", image_cv.shape)


class ModelCV(ModelBase):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        do_copy=True,
    ) -> None:
        super().__init__(
            model_prefix, vdsp_config, batch_size, device_id, hw_config, do_copy
        )

    def process(
        self, input: Union[List[np.ndarray], List[vsx.Image], np.ndarray, vsx.Image]
    ):
        if isinstance(input, list):
            if isinstance(input[0], np.ndarray):
                return self.process(
                    [cv_rgb_image_to_vastai(x, self.device_id_) for x in input]
                )
            else:
                return self.process_impl(input)
        return self.process([input])[0]

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height = input_shape[-2]
        width = input_shape[-1]
        dummy = np.zeros((height, width, 3), dtype=dtype)
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            vacc_dummy = vsx.create_image(
                dummy, vsx.ImageFormat.BGR_INTERLEAVE, width, height, self.device_id_
            )
            return [vacc_dummy] * batch_size

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        # return [[vsx.as_numpy(o) for o in out] for out in outputs]
        return [vsx.as_numpy(out[0]) for out in outputs]


class MobileVit(ModelCV):
    def __init__(
        self, model_prefix, vdsp_config, batch_size=1, device_id=0, hw_config=""
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)
        self.model_input_height, self.model_input_width = self.model_.input_shape[0][
            -2:
        ]
        self.resize_height = int(256.0 / 224 * self.model_input_height)
        self.fusion_op_ = None
        for op in self.preproc_ops_:
            if op.op_type >= 100:
                self.fusion_op_ = op.cast_to_buildin_operator()
                break
        assert self.fusion_op_ is not None, "Can't find fusion op in vdsp op json file"

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height = input_shape[-2]
        width = input_shape[-1]
        dummy = np.zeros((height, width, 3), dtype=dtype)
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            device_dummy = vsx.create_image(
                dummy, vsx.ImageFormat.BGR_INTERLEAVE, width, height, self.device_id_
            )
            return [device_dummy] * batch_size

    def compute_size(self, img_w, img_h, size):
        if isinstance(size, int):
            size_h, size_w = size, size
        elif len(size) < 2:
            size_h, size_w = size[0], size[0]
        else:
            size_h, size_w = size[-2:]

        r = max(size_w / img_w, size_h / img_h)

        new_w = int(r * img_w)
        new_h = int(r * img_h)
        return (new_w, new_h)

    def process_impl(self, inputs):
        outputs = []
        for input in inputs:
            resize_width, resize_height = self.compute_size(
                input.width, input.height, self.resize_height
            )
            left = (resize_width - self.model_input_width) // 2
            top = (resize_height - self.model_input_height) // 2
            self.fusion_op_.set_attribute(
                {
                    attr.IIMAGE_WIDTH: input.width,
                    attr.IIMAGE_HEIGHT: input.height,
                    attr.IIMAGE_WIDTH_PITCH: input.width,
                    attr.IIMAGE_HEIGHT_PITCH: input.height,
                    attr.RESIZE_WIDTH: resize_width,
                    attr.RESIZE_HEIGHT: resize_height,
                    attr.CROP_X: left,
                    attr.CROP_Y: top,
                }
            )
            model_outs = self.stream_.run_sync([input])[0]
            outputs.append(vsx.as_numpy(model_outs[0]).astype(np.float32))
        return outputs



def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vastpipe/data/models/mobilevits-fp16-none-1_3_224_224-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        help="hw-config file of the model suite",
        default="",
    )
    parser.add_argument(
        "--vdsp_params",
        default="../../../../data/configs/mobilevit_rgbplanar.json",
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
        "-s",
        "--shape",
        help="model input shape",
    )
    parser.add_argument(
        "--iterations",
        default=1024,
        type=int,
        help="iterations count for one profiling",
    )
    parser.add_argument(
        "--queue_size",
        default=0,
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
    hw_config = args.hw_config
    vdsp_params = args.vdsp_params
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
        model = MobileVit(
            model_prefix, vdsp_params, batch_size=batch_size, device_id=device_id
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
