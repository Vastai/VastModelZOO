import ctypes
import glob
import json
import time
from queue import Queue
from threading import Event, Thread
from typing import Dict, Generator, Iterable, Union

import cv2 as cv
import numpy as np
import vacl_stream
import vaststream


class FeatureExtract:
    def __init__(
        self,
        model_info: Union[str, Dict[str, str]],
        vdsp_params_info: str,
        device_id: int = 0,
        batch_size: int = 1,
    ) -> None:
        if isinstance(model_info, str):
            with open(model_info) as f:
                model_info = json.load(f)

        self.device_id = device_id
        self.input_id = 0
        self.vast_stream = vaststream.vast_stream()
        self.input_dict = {}
        self.event_dict = {}
        self.result_dict = {}

        balance_mode = 0

        def callback(output_description, ulOutPointerArray, ulArraySize, user_data_ptr):
            user_data = ctypes.cast(user_data_ptr, ctypes.POINTER(vacl_stream.StreamInfo))
            input_id = output_description.contents.input_id

            device_ddr = self.input_dict.pop(input_id)
            self.vast_stream.free_data_on_device(device_ddr, self.device_id)

            model_name = user_data.contents.model_name
            stream_output_list = self.vast_stream.stream_get_stream_output(model_name, ulOutPointerArray, ulArraySize)
            heatmap = np.squeeze(stream_output_list[0])
            _, shape = self.vast_stream.get_output_shape_by_index(model_name, 0)
            ndims = shape.ndims
            heatmap_shape = []
            for i in range(ndims):
                heatmap_shape.append(shape.shapes[i])
            self.result_dict[input_id] = (heatmap_shape,heatmap)
            self.event_dict[input_id].set()

        self.callback = vaststream.output_callback_type(callback)

        self.stream = vacl_stream.create_vaststream(
            device_id,
            vdsp_params_info,
            model_info,
            self.callback,
            balance_mode,
            batch_size,
        )

    def __start_extract(self, image: Union[str, np.ndarray]) -> int:
        if isinstance(image, str):
            image = cv.imread(image, cv.IMREAD_COLOR)
            image = np.stack(cv.split(cv.cvtColor(image, cv.COLOR_BGR2RGB)))
        # image = np.load(image)

        assert len(image.shape) == 3
        # image = image[:,:,:-1]

        c, height, width = image.shape
        assert c == 3
        image_size = int(height * width * c)
        device_ddr = self.stream.copy_data_to_device(image, image_size)

        input_id = self.input_id

        self.input_dict[input_id] = device_ddr
        self.event_dict[input_id] = Event()
        self.stream.run_stream_dynamic([device_ddr], (height, width), input_id)
        self.input_id += 1

        return input_id

    def extract(self, image: Union[str, np.ndarray]) -> str:
        input_id = self.__start_extract(image)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]
        return result

    def extract_batch(self, images: Iterable[Union[str, np.ndarray]]) -> Generator[str, None, None]:
        queue = Queue(20)

        def input_thread():
            for image in images:
                input_id = self.__start_extract(image)
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
