import ctypes
import json
from queue import Queue
from threading import Event, Thread
from typing import Dict, Generator, Iterable, List, Union

import cv2 as cv
import numpy as np
import vacl_stream
import vaststream

class Detector:
    def __init__(
        self,
        model_info: Union[str, Dict[str, str]],
        vdsp_params_info: str,
        classes: Union[str, List[str]],
        device_id: int = 0,
        batch_size: int = 1,
        threashold: float = 0.01
    ) -> None:
        if isinstance(model_info, str):
            with open(model_info) as f:
                model_info_dict = json.load(f)

        if isinstance(vdsp_params_info, str):
            with open(vdsp_params_info) as f:
                vdsp_params_info_dict = json.load(f)

        if isinstance(classes, str):
            with open(classes) as f:
                classes = [cls.strip() for cls in f.readlines()]

        self.model_size = vdsp_params_info_dict["oimage_width"]
        self.classes = classes
        self.device_id = device_id
        self.input_id = 0
        self.threashold = threashold
        self.vast_stream = vaststream.vast_stream()

        self.input_dict = {}
        self.event_dict = {}
        self.result_dict = {}
        balance_mode = 0


        def callback(output_description, ulOutPointerArray, ulArraySize, user_data_ptr):
            user_data = ctypes.cast(user_data_ptr, ctypes.POINTER(vacl_stream.StreamInfo))
            input_id = output_description.contents.input_id

            device_ddr, height, width = self.input_dict.pop(input_id)
            # free input image data on Device
            self.vast_stream.free_data_on_device(device_ddr, self.device_id)
            # get stream output result
            model_name = user_data.contents.model_name
            stream_output_list = self.vast_stream.stream_get_stream_output(model_name, ulOutPointerArray, ulArraySize)
            # get nms results
            self.result_dict[input_id] = []
            self.post_processing(self.classes, input_id, height, width, stream_output_list)
            self.event_dict[input_id].set()

        self.callback = vaststream.output_callback_type(callback)

        self.stream = vacl_stream.create_vaststream(
            device_id,
            vdsp_params_info,
            model_info_dict,
            self.callback,
            balance_mode,
            batch_size,
            **{"yolo_size": self.model_size}
        )

    def post_processing(self, class_list, input_id, height, width, stream_output_list):
        FROM_PYTORCH = True
        # in this demo/test case, 1 input --> 1 output
        stream_ouput_data = stream_output_list[0]
        # yolov3: 3 outputs, classes = 100? 300 when pytorch.
        box_ids = stream_ouput_data[0]
        num_outputs = len(box_ids)
        box_scores = stream_ouput_data[1]
        # (num_outputs, 4): xmin, ymin, xmax, ymax
        box_coords = stream_ouput_data[2]
        box_coords = np.reshape(box_coords, (num_outputs, 4))

        # NOTE: extra threashold button -> for hign performance can remove
        indexes = np.where(np.logical_and(box_scores > self.threashold, box_ids != -1))
        box_ids = box_ids[indexes]
        box_scores = box_scores[indexes]
        box_coords = box_coords[indexes]

        # post processing
        r = min(self.model_size / width, self.model_size / height)
        # r = min(r, 1.0) # scale up is allowed
        unpad_w = int(round(width * r))
        unpad_h = int(round(height * r))
        dw = self.model_size - unpad_w
        dh = self.model_size - unpad_h
        dw /= 2
        dh /= 2
        w = width / unpad_w
        h = height / unpad_h

        box_coords[:, 0] = (box_coords[:, 0] - dw) * w
        box_coords[:, 2] = (box_coords[:, 2] - dw) * w
        box_coords[:, 1] = (box_coords[:, 1] - dh) * h
        box_coords[:, 3] = (box_coords[:, 3] - dh) * h
        if FROM_PYTORCH:
            box_coords[:, 2] = box_coords[:, 2] - box_coords[:, 0]
            box_coords[:, 3] = box_coords[:, 3] - box_coords[:, 1]
        else:
            box_coords[:, 2] = box_coords[:, 2] - box_coords[:, 0] + 1
            box_coords[:, 3] = box_coords[:, 3] - box_coords[:, 1] + 1

        # NOTE: for hign performance can remove box_labels
        box_labels = [self.classes[int(cls.astype('float32'))] for cls in box_ids]
        self.result_dict[input_id] = [box_ids, box_scores, box_coords, box_labels]

    def _start_detection(self, image: Union[str, np.ndarray]):
        if isinstance(image, str):
            image = cv.imread(image, cv.IMREAD_COLOR)
            image = np.stack(cv.split(cv.cvtColor(image, cv.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3

        image_size = int(height * width * c)

        device_ddr = self.stream.copy_data_to_device(image, image_size)

        input_id = self.input_id

        self.input_dict[input_id] = device_ddr, height, width
        self.event_dict[input_id] = Event()
        self.stream.run_stream_dynamic([device_ddr], (height, width), input_id)
        self.input_id += 1

        return input_id

    def detection(self, image: Union[str, np.ndarray]):
        input_id = self._start_detection(image)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]
        return result

    def detection_batch(self, images: Iterable[Union[str, np.ndarray]]) -> Generator[str, None, None]:
        # TODO: adaptive queue number
        queue = Queue(20)

        def input_thread():
            for image in images:
                input_id = self._start_detection(image)
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
    detector = Detector(
        model_info="./vacl_info/model_info_yolov3.json",
        vdsp_params_info="./vacl_info/vdsp_params_yolov3_letterbox_rgb.json",
        classes="/home/lance/workspace/VastDeploy/data/eval/coco2id.txt",
        device_id=0,
        batch_size=1,
    )

    # Test one image from path
    image_path = "/home/lance/workspace/VastDeploy/data/test/od/0.jpg"
    result = detector.detection(image_path)
    print(f"{image_path} => {result}")

    # Test one image from numpy array
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image = np.stack(cv.split(cv.cvtColor(image, cv.COLOR_BGR2RGB)))
    result = detector.detection(image)
    print(f"{image_path} => {result}")

    # Test multiple images
    images = glob.glob("/home/lance/workspace/VastDeploy/data/eval/det_coco_calib/*.jpg")
    time_begin = time.time()
    results = detector.detection_batch(images)
    for (image, result) in zip(images, results):
        print(f"{image} => {result}")
    time_end = time.time()

    print(
        f"\n{len(images)} images in {time_end - time_begin} seconds, ({len(images) / (time_end - time_begin)} images/second)\n"
    )

