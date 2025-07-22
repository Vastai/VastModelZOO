from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import json
import sys
sys.path.append('./')
from queue import Queue
from threading import Event, Thread
from typing import Dict, Generator, Iterable, List

import cv2
import numpy as np
import torch

import vacl_stream
import vaststream
from vaststream import *

class StreamInfo(ctypes.Structure):
    _fields_ = [
                ("stream_id", ctypes.c_uint32),
                ("device_id", ctypes.c_uint32),
                ("model_op_context", ctypes.c_uint32),
                ("model_name", ctypes.c_wchar_p)
                ]


class NLP:
    def __init__(
        self,
        model_info: str,
        bytes_size: int = 1536,
        device_id: int = 0,
        batch_size: int = 1,
        **kwargs
        ):
        # get model path and hw_config path
        if isinstance(model_info, str):
            with open(model_info) as f:
                model_info = json.load(f)       
        
        self.vacl_stream = vacl_stream.VaclStream()
        self.input_id = 0
        self.__input_dict = {}
        self.__event_dict = {}
        self.__result_dict = {}
        self.balance_mode = 0
        self.device_id = device_id
        self.batch_size = batch_size
        self.bytes_size = bytes_size
        self.vast_stream = vast_stream()

        self.vast_stream.init(self.device_id)
        init_context, stream_id = self.vast_stream.create_stream(self.device_id, self.balance_mode)
        assert stream_id != 0, 'Fail to create stream.'

        self.stream_id = stream_id
        self.init_op_context = init_context

        # create stream and build model 
        op_context_id = self.vast_stream.create_stream_op(self.stream_id, 20001)
        self.vast_stream.invoke_stream_op(op_context_id, self.init_op_context)
        self.vdsp_op_context = op_context_id
        model_name = self._register_model_info(model_info)
        self.model_name = model_name
        model_op_context = self.vast_stream.invoke_run_model_op(self.stream_id, self.vdsp_op_context, model_name)
        self.model_op_context = model_op_context
        self.vast_stream.build_stream(self.stream_id)

        # input_num = self.vast_stream.get_stream_input_data_num(stream_id)
        # input_mem = 1 # on device ddr
        # self.input_desc_p = self.vast_stream.create_input_data_desc(input_num, input_mem)

        def callback(output_description, ulOutPointerArray, ulArraySize, user_data_ptr):
            user_data = ctypes.cast(user_data_ptr, ctypes.POINTER(vacl_stream.StreamInfo))
            input_id = output_description.contents.input_id
            device_ddr = self.__input_dict.pop(input_id)
            if isinstance(device_ddr, list):
                for ddr in device_ddr:
                    free_reuslt = self.vast_stream.free_data_on_device(ddr, self.device_id)
                    if free_reuslt != 0:
                        raise "free error.\n"
            else:
                self.vast_stream.free_data_on_device(device_ddr, self.device_id)

            model_name = user_data.contents.model_name
            stream_output_list = self._get_stream_output(ulOutPointerArray, ulArraySize)
            heatmap = np.squeeze(stream_output_list[0])
            _, shape = self.vast_stream.get_output_shape_by_index(model_name, 0)
            ndims = shape.ndims
            heatmap_shape = []
            for i in range(ndims):
                heatmap_shape.append(shape.shapes[i])
            
            heatmap = heatmap.reshape(heatmap_shape)
            self.__result_dict[input_id] = heatmap
            self.__event_dict[input_id].set()
            print("callback count:", input_id, end=',')

        self.callback_output = vaststream.output_callback_type(callback)
        self.user_data = StreamInfo(self.stream_id, self.device_id, self.model_op_context, self.model_name)
        user_data_ptr = cast(POINTER(StreamInfo)(self.user_data), c_void_p)
        self.vast_stream.subscribe_device_report(self.stream_id, self.callback_output, user_data_ptr)   

    def _register_model_info(self, model_info_dict: Dict):
        assert self.stream_id != 0       
        model_name = model_info_dict.get('model_name')
        model_hw_cfg_file = model_info_dict.get('hw_config_file')
        model_dynamic_json = model_info_dict.get('model_dynamic_json')

        # dynamic model description has higher priority here. 
        if model_dynamic_json != None:
            if self.vast_stream.register_dynamic_model(model_name, model_dynamic_json, self.device_id, model_hw_cfg_file, self.batch_size):
                raise ValueError("Failed to register dynamic model.")
        else:
            model_lib_path = model_info_dict.get('model_lib_path')
            model_graph_path = model_info_dict.get('model_graph_path')
            model_params_path = model_info_dict.get('model_params_path')
            if self.vast_stream.register_static_model(model_name, model_lib_path, model_graph_path, model_params_path, self.device_id, model_hw_cfg_file, self.batch_size):
                raise ValueError("Failed to register model.")
        return model_name
    
    def _get_stream_output(self, ulOutPointerArray, ulArraySize, **kwargs):
        output_count_per_batch = self.vast_stream.get_output_num_per_batch(self.model_name)
        assert output_count_per_batch != 0
        count = int(ulArraySize / output_count_per_batch)
        result_list = []
        index = 0
        for i in range(count):
            data_list = []
            for j in range(output_count_per_batch):
                size = self.vast_stream.get_output_size_by_index(self.model_name, j) # bytes
                data = np.zeros(int(size / 2), dtype=np.float16) # fp16
                output_data = data.ctypes.data_as(ctypes.c_void_p)
                self.vast_stream.get_model_output(self.model_name, j, c_uint64(ulOutPointerArray[index]), output_data)
                data_list.append(data)
                index += 1
            result_list.append(data_list)    
        return result_list

    def _copy_to_device(self, data, bytes_size):
        host_addr = data.ctypes.data_as(ctypes.c_void_p)
        device_addr = self.vast_stream.copy_data_to_device(host_addr, bytes_size, self.device_id)
        return device_addr
    
    def _run_stream(self, inputs: List[np.ndarray]):
        # copy data to device and get input addr on device
        addrs_list = [self._copy_to_device(input, self.bytes_size) for input in inputs]
        data_array_size = len(addrs_list)
        data_array = (c_uint64 * data_array_size) (*addrs_list)     
        input_id = self.input_id
        self.__input_dict[input_id] = addrs_list
        self.__event_dict[input_id] = Event()
        print(" Testing:", input_id)

        # run model
        ret = self.vast_stream.run_stream(self.stream_id, self.input_id, data_array, len(addrs_list))
        self.input_id += 1
        return input_id

    def run(self, inputs: List[np.ndarray]):
        inputs_len = len(inputs)   
        if inputs_len < 1:
            raise ValueError('The number of inputs should be greater than 0.')
        
        input_id = self._run_stream(inputs)
        self.__event_dict[input_id].wait()
        result = self.__result_dict.pop(input_id)
        del self.__event_dict[input_id]
        return result
    
    def run_batch(self, datasets: Iterable[np.ndarray]) -> Generator[str, None, None]:
        datasets_len = self.files_len  if hasattr(self, 'files_len') else 0
        if datasets_len < 1:
            raise ValueError('The number of data sets should be greater than 0.')   
        queue = Queue(20)

        def input_thread():
            for data in datasets:
                input_id = self._run_stream(data)
                queue.put(input_id)
            queue.put(None)

        thread = Thread(target=input_thread)
        thread.start()

        while True:
            input_id = queue.get()
            if input_id is None:
                break
            self.__event_dict[input_id].wait()
            result = self.__result_dict.pop(input_id)
            del self.__event_dict[input_id]
            yield result


    def get_datasets(self, datasets_dir: str):
        npz_datalist_fr = open(datasets_dir, 'r')
        npz_datalist = npz_datalist_fr.readlines()
        
        self.files_len = len(npz_datalist)
        if self.files_len == 0:
            raise ValueError('dataset files is None.')

        # def dataset_loader():
        #     for index, data_path in enumerate(npz_datalist):
        #         data_path = data_path.strip()
        #         npz_data = np.load(data_path)
        #         inputs_ids_1 = npz_data[npz_data.files[0]]
        #         segment_ids_1 = npz_data[npz_data.files[1]]
        #         input_mask_1 = npz_data[npz_data.files[2]]
                
        #         yield [inputs_ids_1, segment_ids_1, input_mask_1]
        
        def dataset_loader():
            for index, data_path in enumerate(npz_datalist):
                data = []
                data_path = data_path.strip()
                npz_data = np.load(data_path)
                
                for k, v in npz_data.items():
                    data.append(v)
                
                yield data
                
        return dataset_loader