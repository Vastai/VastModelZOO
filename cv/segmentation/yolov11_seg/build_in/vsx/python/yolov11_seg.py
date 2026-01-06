# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import time
import queue
import threading
import concurrent.futures
from easydict import EasyDict as edict
import vaststreamx as vsx
import ctypes
from typing import Union, List

attr = vsx.AttrKey

def imagetype_to_vsxformat(imagetype):
    if imagetype == 0:
        return vsx.ImageFormat.YUV_NV12
    elif imagetype == 5000:
        return vsx.ImageFormat.RGB_PLANAR
    elif imagetype == 5001:
        return vsx.ImageFormat.BGR_PLANAR
    elif imagetype == 5002:
        return vsx.ImageFormat.RGB_INTERLEAVE
    elif imagetype == 5003:
        return vsx.ImageFormat.BGR_INTERLEAVE
    elif imagetype == 5004:
        return vsx.ImageFormat.GRAY
    else:
        assert False, f"Unrecognize image type {imagetype}"

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
    
class ModelBase:
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        output_type=vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE,
    ) -> None:
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

    def get_fusion_op_iimage_format(self):
        for op in self.preproc_ops_:
            if op.op_type >= 100:
                buildin_op = op.cast_to_buildin_operator()
                if "IIMAGE_FORMAT" in list(buildin_op.attributes.keys()):
                    imagetype = buildin_op.get_attribute(attr.IIMAGE_FORMAT)
                    return imagetype_to_vsxformat(imagetype)
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


class ModelCV(ModelBase):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        output_type=vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE,
    ) -> None:
        super().__init__(
            model_prefix, vdsp_config, batch_size, device_id, hw_config, output_type
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
        return [[vsx.as_numpy(o) for o in out] for out in outputs]


class CustomOpBase:
    def __init__(self, op_name, elf_file, device_id):
        self.device_id_ = device_id
        vsx.set_device(device_id)
        self.custom_op_ = vsx.CustomOperator(op_name=op_name, elf_file_path=elf_file)


class image_shape_layout_t(ctypes.Structure):
    _fields_ = [
        ("height", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("h_pitch", ctypes.c_int32),
        ("w_pitch", ctypes.c_int32),
    ]


class yolov8_seg_op_t(ctypes.Structure):
    _fields_ = [
        ("model_in_shape", image_shape_layout_t),
        ("model_out_shape", image_shape_layout_t),
        ("origin_image_shape", image_shape_layout_t),
        ("k", ctypes.c_uint32),
        ("retina_masks", ctypes.c_uint32),
        ("max_detect_num", ctypes.c_uint32),
    ]


class Yolov8SegPostProcOp(CustomOpBase):
    def __init__(self, op_name, elf_file, device_id=0, retina_masks=True) -> None:
        super().__init__(op_name=op_name, elf_file=elf_file, device_id=device_id)
        self.custom_op_.set_callback_info(
            [(1, int(376 * 1.5), 500)] * 5, [(1, int(376 * 1.5), 500)] * 6
        )
        self.retina_masks_ = retina_masks

    def process(self, input_tensors, model_input_shape, image_shape):
        inputs = [input.clone(self.device_id_) for input in input_tensors]
        mask_shape = inputs[4].shape
        classes_shape = inputs[0].shape
        model_in_height, model_in_width = model_input_shape[-2:]
        model_out_height, model_out_width = mask_shape[-2:]
        image_height, image_width = image_shape[-2:]
        max_detect_num = classes_shape[1]
        mask_ch_num = 32  # Don't change this parameter

        op_conf = yolov8_seg_op_t()
        op_conf.model_in_shape.height = model_in_height
        op_conf.model_in_shape.width = model_in_width
        op_conf.model_in_shape.h_pitch = model_in_height
        op_conf.model_in_shape.w_pitch = model_in_width

        op_conf.model_out_shape.height = model_out_height
        op_conf.model_out_shape.width = model_out_width
        op_conf.model_out_shape.h_pitch = model_out_height
        op_conf.model_out_shape.w_pitch = model_out_width
        op_conf.origin_image_shape.height = image_height
        op_conf.origin_image_shape.width = image_width
        op_conf.origin_image_shape.h_pitch = image_height
        op_conf.origin_image_shape.w_pitch = image_width

        op_conf.k = mask_ch_num
        op_conf.retina_masks = 1 if self.retina_masks_ else 0
        op_conf.max_detect_num = max_detect_num

        op_conf_size = ctypes.sizeof(yolov8_seg_op_t)

        mask_out_h = model_in_height
        mask_out_w = model_in_width
        if self.retina_masks_:
            mask_out_h = image_height
            mask_out_w = image_width

        buffer_size = (max_detect_num + 3) * max(
            model_in_width * model_in_height, image_height * image_width
        )
        config_bytes = ctypes.string_at(ctypes.byref(op_conf), op_conf_size)
        outputs = self.custom_op_.run_sync(
            tensors=inputs,
            config=config_bytes,
            output_info=[
                ([max_detect_num], vsx.TypeFlag.FLOAT16),
                ([max_detect_num], vsx.TypeFlag.FLOAT16),
                ([max_detect_num, 4], vsx.TypeFlag.FLOAT16),
                ([max_detect_num, mask_out_h, mask_out_w], vsx.TypeFlag.UINT8),
                ([2], vsx.TypeFlag.UINT32),
                ([buffer_size], vsx.TypeFlag.UINT8),
            ],
        )
        res = outputs[:5]
        return res


class Yolov11Segmenter(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        elf_file,
        batch_size=1,
        device_id=0,
        hw_config="",
    ) -> None:
        super().__init__(
            model_prefix,
            vdsp_config,
            batch_size,
            device_id,
            hw_config,
            output_type=vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_HOST,
        )
        self.post_proc_op_ = Yolov8SegPostProcOp("yolov8_seg_op", elf_file, device_id)

    def process_impl(self, input):
        model_outs = self.stream_.run_sync(input)
        outputs = []
        for inp, mod_out in zip(input, model_outs):
            op_outs = self.post_process(mod_out, inp.width, inp.height)
            num = vsx.as_numpy(op_outs[4])[0]
            if num > 0:
                outs = [vsx.as_numpy(out) for out in op_outs]
                outputs.append(outs)
            else:
                outputs.append([])
        return outputs

    def post_process(self, fp16_tensors, image_width, image_height):
        return self.post_proc_op_.process(
            fp16_tensors, self.model_.input_shape[0], [image_height, image_width]
        )
    

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
                if context.left > 0:
                    fut = queue_futs.get(timeout=0.01)
                    fut.result()
                    tock = time.time()
                    if self.long_time_test_ is False:
                        tocks.append(tock)
                    context.left -= 1
                else:
                    time.sleep(0.00001)

        cunsume_thread = threading.Thread(
            target=cunsume_thread_func, args=(context, queue_futs, tocks)
        )
        cunsume_thread.start()
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while self.iters_left_ > 0 or self.long_time_test_:
                tick = time.time()
                fut = executor.submit(self.process_async, self.models_[idx], infer_data)
                queue_futs.put(fut)
                self.iters_left_ -= 1
                context.left += 1
                if self.long_time_test_ is False:
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

        if self.long_time_test_:
            while True:
                self.models_[idx].process(infer_data)

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
