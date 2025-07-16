import os

import numpy as np
import cv2
import argparse
import vaststreamx as vsx
from typing import Union, List
import glob
from tqdm import tqdm

def load_labels(file):
    with open(file) as fin:
        return [line.strip() for line in fin.readlines()]

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
        self, model_prefix, vdsp_config, batch_size=1, device_id=0, hw_config="",do_copy=True
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


class DetrModel(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        threshold=0.2,
        hw_config="",
    ) -> None:
        super().__init__(
            model_prefix, vdsp_config, batch_size, device_id, hw_config, do_copy=False
        )
        self.threshold_ = threshold

    def set_threshold(self, threshold):
        self.threshold_ = threshold

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [
            self.post_process(output, input[i].width, input[i].height)
            for i, output in enumerate(outputs)
        ]

    def post_process(self, fp16_tensors, image_width, image_height):
        def np_softmax(x, axis=-1):
            """Compute softmax values for each sets of scores in x along axis."""
            e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return e_x / e_x.sum(axis=axis, keepdims=True)

        def np_max(x, axis=None, keepdims=False):
            max_val = np.max(x, axis=axis, keepdims=keepdims)
            max_idx = np.argmax(x, axis=axis)
            if keepdims:
                max_idx = np.expand_dims(max_idx, axis)
            return max_val, max_idx

        def box_cxcywh_to_xyxy(x):
            x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return np.stack(b, axis=-1)

        def unpad(x, dw, dh, r):
            x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
            b = [(x1 - dw) / r, (y1 - dh) / r, (x2 - dw) / r, (y2 - dh) / r]
            return np.stack(b, axis=-1)

        model_height, model_width = self.model_.input_shape[0][-2:]
        out_logits = vsx.as_numpy(fp16_tensors[0]).astype(np.float32)
        out_bbox = vsx.as_numpy(fp16_tensors[1]).astype(np.float32)
        # print(out_logits.shape)
        # print(out_bbox.shape)

        prob = np_softmax(out_logits, -1)
        scores, labels = np_max(prob[..., :-1], axis=-1)

        boxes = box_cxcywh_to_xyxy(out_bbox)
        scale_fct = [[[model_width, model_height, model_width, model_height]]]
        boxes = boxes * scale_fct
        r = min(model_width / image_width, model_height / image_height)
        unpad_w = image_width * r
        unpad_h = image_height * r
        dw = (model_width - unpad_w) / 2
        dh = (model_height - unpad_h) / 2
        boxes = unpad(boxes, dw, dh, r).squeeze()

        data_count = len(scores)
        result_np = np.zeros((data_count, 6), dtype=np.float32) - 1
        n = 0
        for i in range(data_count):
            score = scores[i]
            if score >= self.threshold_:
                box = boxes[i]
                result_np[n][0] = labels[i]
                result_np[n][1] = score
                result_np[n][2] = box[0]
                result_np[n][3] = box[1]
                result_np[n][4] = box[2] - box[0]
                result_np[n][5] = box[3] - box[1]
                n += 1
        return result_np


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/path/to/detr_model/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--vdsp_params",
        default="../vacc_code/vdsp_params/facebook-detr-vdsp_params.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="device id to run",
    )
    parser.add_argument(
        "--label_file",
        default="/path/to/coco91.txt",
        help="label file",
    )
    parser.add_argument(
        "--input_file",
        default="/path/to/dog.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="",
        help="output file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="dataset filename list",
    )
    parser.add_argument(
        "--file_path",
        default="",
        help="dataset root",
    )
    parser.add_argument(
        "--save_dir",
        default="",
        help="dataset output folder path",
    )
    args = parser.parse_args()
    return args


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # NOTE
    # 此处暂时处理为方形
    # if auto:  # minimum rectangle
    #     dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # elif scaleFill:  # stretch
    #     dw, dh = 0.0, 0.0
    #     new_unpad = (new_shape[1], new_shape[0])
    #     ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

if __name__ == "__main__":
    args = argument_parser()
    labels = load_labels(args.label_file)
    batch_size = 1
    model = DetrModel(args.model_prefix, args.vdsp_params, batch_size, args.device_id)
    model.set_threshold(args.threshold)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if os.path.isfile(args.file_path):
        image = cv2.imread(args.input_file)
        assert image is not None, f"Failed to read {args.input_file}"
        objects = model.process(image)
        print("Detection objects:")
        for obj in objects:
            if obj[1] >= 0:
                bbox = [int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])]
                print(
                    f"Object class: {labels[int(obj[0])]}, score: {obj[1]}, bbox: {bbox}"
                )
            else:
                break
        if args.output_file != "":
            for obj in objects:
                if obj[1] >= 0:
                    cv2.rectangle(
                        image,
                        pt1=(int(obj[2]), int(obj[3])),
                        pt2=(int(obj[2] + obj[4]), int(obj[3] + obj[5])),
                        color=(0, 0, 255),
                        thickness=1,
                    )
                else:
                    break
            cv2.imwrite(args.output_file, image)
    else:
        filelist = []
        images = glob.glob(os.path.join(args.file_path, "*"))
        for image_file in tqdm(images):
            fullname = image_file
            image = cv2.imread(fullname)
            objects = model.process(image)
            base_name, _ = os.path.splitext(os.path.basename(image_file))
            outfile = open(os.path.join(args.save_dir, base_name + ".txt"), "w")
            print(f"{image_file} detection objects:")
            for obj in objects:
                if obj[1] >= 0:
                    bbox = [int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])]
                    # bbox_x = int(obj[2])
                    # bbox_y = int(obj[3])
                    # bbox_w = int(obj[4])
                    # bbox_h = int(obj[5])
                    # cv2.rectangle(image, [bbox_x, bbox_y, bbox_w, bbox_h], color=(0, 255, 0), thickness=2)
                    print(f"Object class: {labels[int(obj[0])]}, score: {obj[1]}, bbox: {bbox[0]}, {bbox[1]}, {bbox[0] + bbox[2]}, {bbox[1] + bbox[3]}")
                    outfile.write(f"{labels[int(obj[0])]} {obj[1]} {int(obj[2])} {int(obj[3])} {int(obj[2]+obj[4])} {int(obj[3]+obj[5])}\n")
                else:
                    break
            
            # cv2.imwrite("vsx_result.jpg", image)
            outfile.close()


