import os
import cv2
import glob
import json
import torch
import ctypes
import numpy as np

from PIL import Image
from tqdm.contrib import tzip
from torch.nn import functional as F

from queue import Queue
from threading import Event, Thread
from typing import Dict, Generator, Iterable, Union

import vacl_stream
import vaststream


class Classifier:
    def __init__(
        self,
        model_info: Union[str, Dict[str, str]],
        vdsp_params_info: str,
        device_id: int = 1,
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
            print("5")
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
            print("2")
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
            image = cv2.imread(image, cv2.IMREAD_COLOR)
            image = np.stack(cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        image_size = int(height * width * c)
        print("3")
        device_ddr = self.stream.copy_data_to_device(image, image_size)

        input_id = self.input_id

        self.input_dict[input_id] = device_ddr
        self.event_dict[input_id] = Event()
        self.stream.run_stream_dynamic([device_ddr], (height, width), input_id)
        print("4")
        self.input_id += 1

        return input_id

    def classify(self, image: Union[str, np.ndarray]) -> str:
        input_id = self.__start_classify(image)
        print("1")
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]
        return result

    def classify_batch(self, images: Iterable[Union[str, np.ndarray]]) -> Generator[str, None, None]:
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


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


if __name__ == '__main__':
    model_info = "segmentation/refinenet/vacc_code/runstream/legacy_1.x/model_info/model_info.json"
    vdsp_params_info ="segmentation/refinenet/vacc_code/runstream/legacy_1.x/model_info/vdsp_params.json"
    device_id  = 0
    batch_size = 1

    classifier = Classifier(
        model_info=model_info,
        vdsp_params_info=vdsp_params_info,
        device_id=device_id,
        batch_size=batch_size,
    )


    data_dir = "/home/simplew/dataset/seg/VOCdevkit/VOC2012/JPEGImages_val"
    gt_dir = "/home/simplew/dataset/seg/VOCdevkit/VOC2012/SegmentationClass"

    result_dir = "./runstream_result"
    os.makedirs(result_dir, exist_ok=True)

    import sys
    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + '../../..')
    from source_code.drsleep.score import SegmentationMetric
    
    input_size = 500
    metric = SegmentationMetric(21)

    image_files = glob.glob(os.path.join(data_dir, "*.jpg"))

    results = classifier.classify_batch(image_files)
    for (image_path, result) in tzip(image_files, results):
        ori_image = Image.open(image_path)

        heatmap = np.reshape(result[1].astype(np.float32),result[0])

        # draw
        tvm_predict = torch.from_numpy(heatmap)
        tvm_predict = F.interpolate(tvm_predict, (input_size, input_size), mode='bilinear')     

        predict = tvm_predict[0].cpu().numpy()
        predict = np.asarray(predict, dtype="float32").transpose(1, 2, 0)

        predict_mask = cv2.resize(predict, ori_image.size, interpolation=cv2.INTER_CUBIC)

        colors = np.loadtxt("segmentation/refinenet/source_code/drsleep/voc2012_colors.txt").astype('uint8')
        color = colorize(predict_mask.argmax(axis=2).astype(np.uint8), colors)
        color.save(os.path.join(result_dir, os.path.basename(image_path)+".png"))

        ########################################################################################################
        # eval
        label_path = os.path.join(gt_dir, os.path.basename(image_path).replace(".jpg", ".png"))
        if not os.path.exists(label_path):
            continue
        gt = Image.open(label_path)
        gt = gt.resize(size=(input_size, input_size))
        target = np.array(gt).astype('int32')
        target[target == 255] = -1
        target = torch.from_numpy(target).long()

        metric.update(tvm_predict, target)
        pixAcc, mIoU = metric.get()
        print("{:s}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(image_path, pixAcc * 100, mIoU * 100))
        ########################################################################################################

"""
refinenet_resnet101-int8-kl_divergence-3_500_500-vacc
validation pixAcc: 94.953, mIoU: 77.418
"""