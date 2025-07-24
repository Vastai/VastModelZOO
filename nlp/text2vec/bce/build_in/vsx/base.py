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
from queue import Queue
from threading import Event, Thread
from typing import Dict, Iterable, List, Union

import numpy as np
import vaststreamx as vsx
import datetime
import time

class EmbeddingX:
    def __init__(
        self,
        model_prefix_path: Union[str, Dict[str, str]],
        device_id: int = 0,
        batch_size: int = 1,
        is_async_infer: bool = False,
        model_output_op_name: str = "",
    ):
        self.device_id = device_id
        self.model_output_op_name = model_output_op_name
        self.input_id = 0 

        self.attr = vsx.AttrKey 
        assert vsx.set_device(self.device_id)==0
        # 构建model，模型三件套目录
        model_path = model_prefix_path 
        self.model = vsx.Model(model_path, batch_size) 
        # 输入预处理op
        self.embedding_op = vsx.Operator(vsx.OpType.BERT_EMBEDDING_OP) 
        # 有以上op时无法载通过vsx.Operator加载vdsp算子
        # self.fusion_op = vsx.Operator.load_ops_from_json_file(vdsp_params_info)[0] 
        # 构建graph
        self.graph = vsx.Graph(do_copy=False)
        self.model_op = vsx.ModelOperator(self.model)
        self.graph.add_operators(self.embedding_op, self.model_op) 

        # 构建stream
        self.infer_stream = vsx.Stream(self.graph, vsx.StreamBalanceMode.ONCE)
        if is_async_infer and len(self.model_output_op_name) > 0:
            self.infer_stream.register_operator_output(self.model_output_op_name, self.model_op)
        else:
            self.infer_stream.register_operator_output(self.model_op) 

        # # 预处理算子输出
        self.infer_stream.build() 

        self.current_id = -1
        self.input_dict = {}
        self.event_dict = {}
        self.result_dict = {}
        # 异步
        self.consumer = Thread(target=self.async_receive_infer)
        self.consumer.start()

        self.input_shape = self.model.input_shape

    def async_receive_infer(self):
        while 1:
            try:
                result = None
                if len(self.model_output_op_name) > 0:
                    result = self.infer_stream.get_operator_output(self.model_output_op_name)
                else:
                    result = self.infer_stream.get_operator_output(self.model_op)
                if result is not None:
                    # pre_process_tensor = self.infer_stream.get_operator_output(self.preprocess_name)
                    # 输出顺序和输入一致
                    self.current_id += 1
                    (input_id,) = self.input_dict[self.current_id]
                    model_output_list = [
                        [vsx.as_numpy(out).astype(np.float32) for out in result[0]]
                    ]
                    self.result_dict[input_id] = []
                    self.post_processing(input_id, model_output_list)
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break

    def post_processing(self, input_id, stream_output_list):
        output_data = stream_output_list[0][0]

        self.result_dict[input_id].append(
            {
                "output": output_data,
            }
        )

    def _run(self, vsx_tensors):
        input_id = self.input_id
        self.input_dict[input_id] = (input_id,)
        self.event_dict[input_id] = Event()

        self.infer_stream.run_async([vsx_tensors])
        self.input_id += 1
        return input_id

    def run_batch(self, datasets: Iterable[List[np.ndarray]]):
        queue = Queue(20)

        def input_thread():
            for data in datasets:
                input_id = self._run(data)
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

    def save(self, out, save_dir, name):
        outputs = {}
        outputs = {f'output_{i}': o['output'] for i, o in enumerate(out)}
        np.savez(os.path.join(save_dir, name), **outputs)

    def finish(self):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()
        print("************/n vsx engine del /n************")

    def dummy_datasets(self):

        def dataset_loader():
            npz_datalist = range(300)
            for index, data_path in enumerate(npz_datalist):
                inputs = [np.ones((1, 512)) for _ in range(6)]
                vsx_tensors = [
                    vsx.from_numpy(np.array(input, dtype=np.int32), self.device_id)
                    for input in inputs
                ]

                yield vsx_tensors

        return dataset_loader

    def __call__(self, inputs):
        # start =  datetime.datetime.now()
        # outputs = self.infer_stream.run_sync(inputs)
        # end =  datetime.datetime.now()
        # running_time = end - start
        # print("time cost:", float(end - start)* 1000.0, "s")
        
        # return np.array([vsx.as_numpy(outputs[i][0]) for i in range(len(outputs))])
        start_time = time.time()
        outputs = self.infer_stream.run_sync(inputs)
        end_time = time.time()
        running_time = end_time - start_time
        # print("time cost:", float(end_time - start_time), "s")
        
        return np.array([vsx.as_numpy(outputs[i][0]) for i in range(len(outputs))])
    
    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        if ctx == "CPU":
            return [
                [np.zeros(shape, dtype=dtype) for shape in input_shape]
            ] * batch_size
        else:
            return [
                [
                    vsx.from_numpy(np.zeros(shape, dtype=dtype), self.device_id)
                    for shape in input_shape
                ]
            ] * batch_size
    def process(
        self,
        input: Union[
            List[List[np.ndarray]],
            List[List[vsx.Tensor]],
            List[np.ndarray],
            List[vsx.Tensor],
        ],
    ):
        if isinstance(input[0], list):
            if isinstance(input[0][0], np.ndarray):
                return self.process(
                    [
                        [
                            vsx.from_numpy(np.array(x), self.device_id)
                            for x in one_input
                        ]
                        for one_input in input
                    ]
                )
            else:
                return self.process_impl(input)
        return self.process([input])[0]
    def process_impl(self, input):
        outputs = self.infer_stream.run_sync(input)
        return [[vsx.as_numpy(o).astype(np.float32) for o in out] for out in outputs]

