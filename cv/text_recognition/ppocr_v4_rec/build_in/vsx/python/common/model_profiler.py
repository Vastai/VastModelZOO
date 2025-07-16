import time
import queue
import threading
import concurrent.futures
from easydict import EasyDict as edict
import numpy as np
import vaststreamx as vsx


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
