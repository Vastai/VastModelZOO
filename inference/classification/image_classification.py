# Copyright (C) 2022-2023 VASTAI Technologies Co., Ltd. All Rights Reserved.
import ctypes
import json
from queue import Queue
from threading import Event, Thread
from typing import Dict, Generator, Iterable, List, Union

import cv2 as cv
import numpy as np

import vacl_stream
import vaststream


class Classifier:
    def __init__(
        self,
        model_info: Union[str, Dict[str, str]],
        vdsp_params_info: str,
        classes: Union[str, List[str]],
        topk: int = 1,
        device_id: int = 0,
        batch_size: int = 1,
    ) -> None:
        if isinstance(model_info, str):
            with open(model_info) as f:
                model_info = json.load(f)

        if isinstance(classes, str):
            with open(classes) as f:
                classes = [cls.strip() for cls in f.readlines()]

        self.model_info = model_info
        self.device_id = device_id
        self.classes = classes
        self.input_id = 0
        self.topk = topk
        self.vast_stream = vaststream.vast_stream()
        self.input_dict = {}
        self.event_dict = {}
        self.result_dict = {}

        balance_mode = 0

        def callback(output_description, ulOutPointerArray, ulArraySize,
                     user_data_ptr):
            user_data = ctypes.cast(user_data_ptr,
                                    ctypes.POINTER(vacl_stream.StreamInfo))
            input_id = output_description.contents.input_id

            device_ddr = self.input_dict.pop(input_id)
            self.vast_stream.free_data_on_device(device_ddr, self.device_id)

            model_name = user_data.contents.model_name
            stream_output_list = self.vast_stream.stream_get_stream_output(
                model_name, ulOutPointerArray, ulArraySize)
            data_list = np.squeeze(stream_output_list[0])
            ind = data_list.argsort()[-(self.topk):][::-1]
            classes = [self.classes[i] for i in ind]
            scores = data_list[ind]
            self.result_dict[input_id] = (ind, classes, scores)
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

    def __start_classify(self, image: Union[str, np.ndarray]) -> int:
        if isinstance(image, str):
            image = cv.imread(image, cv.IMREAD_COLOR)
            image = np.stack(cv.split(cv.cvtColor(image, cv.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
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

    def classify(self, image: Union[str, np.ndarray]) -> str:
        input_id = self.__start_classify(image)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]
        return result

    def classify_batch(
        self,
        images: Iterable[Union[str,
                               np.ndarray]]) -> Generator[str, None, None]:
        queue = Queue(20)

        def input_thread():
            for image in images:
                input_id = self.__start_classify(image)
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


if __name__ == '__main__':
    import glob
    import time
    classifier = Classifier(
        model_info='./models/model_info_resnet50.json',
        vdsp_params_info='./models/vdsp_params_resnet50_rgb.json',
        classes='./datasets/res_test2/class2id.txt',
        device_id=0,
        batch_size=1,
    )

    # Test one image from path
    image_path = './datasets/res_test2/normal/ok_test_14.bmp'
    result = classifier.classify(image_path)
    print(f'{image_path} => {result}')

    # Test one image from numpy array
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image = np.stack(cv.split(cv.cvtColor(image, cv.COLOR_BGR2RGB)))
    result = classifier.classify(image)
    print(f'{image_path} => {result}')

    # Test multiple images
    images = glob.glob('./datasets/res_test2/**/*.bmp', recursive=True)
    time_begin = time.time()
    results = classifier.classify_batch(images)
    for (image, result) in zip(images, results):
        print(f'{image} => {result}')
    time_end = time.time()

    print(
        f'\n{len(images)} images in {time_end - time_begin} seconds, ({len(images) / (time_end - time_begin)} images/second)\n'
    )
