# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import vaststreamx as vsx
from tqdm import tqdm
import numpy as np
import cv2
import argparse
from yolov11_seg import Yolov11Segmenter

def load_labels(file):
    with open(file) as fin:
        return [line.strip() for line in fin.readlines()]

def cv_bgr888_to_nv12(bgr888):
    yuv_image = cv2.cvtColor(bgr888, cv2.COLOR_BGR2YUV_I420)
    height, width = bgr888.shape[:2]
    y = yuv_image[:height, :]
    u = yuv_image[height : height + height // 4, :]
    v = yuv_image[height + height // 4 :, :]
    u = np.reshape(u, (height // 2, width // 2))
    v = np.reshape(v, (height // 2, width // 2))
    uv_plane = np.empty((height // 2, width), dtype=np.uint8)
    uv_plane[:, 0::2] = u
    uv_plane[:, 1::2] = v
    yuv_nv12 = np.concatenate((y, uv_plane), axis=0)
    return yuv_nv12
    
def cv_bgr888_to_vsximage(bgr888, vsx_format, device_id):
    h, w = bgr888.shape[:2]
    if vsx_format == vsx.ImageFormat.BGR_INTERLEAVE:
        res = bgr888
    elif vsx_format == vsx.ImageFormat.BGR_PLANAR:
        res = np.array(bgr888).transpose(2, 0, 1)
    elif vsx_format == vsx.ImageFormat.RGB_INTERLEAVE:
        res = cv2.cvtColor(bgr888, cv2.COLOR_BGR2RGB)
    elif vsx_format == vsx.ImageFormat.RGB_PLANAR:
        res = np.array(cv2.cvtColor(bgr888, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    elif vsx_format == vsx.ImageFormat.YUV_NV12:
        res = cv_bgr888_to_nv12(bgr888=bgr888)
    else:
        assert False, f"Unsupport format:{vsx_format}"
    return vsx.create_image(
        res,
        vsx_format,
        w,
        h,
        device_id,
    )


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/path/to/models/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="/path/to/ultralytics-yolov11n_seg-vdsp_params.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--elf_file",
        default="/path/to/elf/yolov8_seg_post_proc",
        help="input file",
    )
    parser.add_argument(
        "--label_file",
        default="/path/to/coco.txt",
        help="label file",
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
        help="detection threshold",
    )
    parser.add_argument(
        "--input_file",
        default="../../../data/images/cycling.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="./infer_result.jpg",
        help="output file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="input dataset image list",
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="input dataset root",
    )
    parser.add_argument(
        "--dataset_output_folder",
        default="",
        help="dataset output folder",
    )
    args = parser.parse_args()
    return args


colors = [
    [62, 140, 230],
    [255, 85, 0],
    [255, 170, 0],
    [255, 0, 85],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 170],
    [255, 0, 255],
    [255, 85, 255],
    [255, 170, 255],
    [0, 255, 255],
    [85, 255, 255],
    [170, 255, 255],
]


def save_mask(tensors, orgin_image, output_image, labels, threshold):
    classes, scores, boxes = tensors[:3]
    det_count = tensors[4][0]
    masks = tensors[3]
    mask_h, mask_w = masks.shape[-2:]
    cvmask = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
    for i in range(det_count):
        if scores[i] < threshold:
            continue
        m = masks[i][:, :, np.newaxis]
        c = i % len(colors)
        msk = m * colors[c]
        cvmask += msk.astype(np.uint8)

    cvorigin = cv2.imread(orgin_image)
    cvorigin = cvorigin * 0.5 + cvmask * 0.5
    for i in range(det_count):
        if scores[i] < threshold:
            continue
        color = colors[i % len(colors)]
        box = boxes[i].astype(int)
        cls_scr = f"{labels[int(classes[i])]},{scores[i]:.2f}"
        cv2.rectangle(cvorigin, (box[0], box[1]), (box[2], box[3]), color)
        font = cv2.FONT_HERSHEY_SIMPLEX
        left, top = box[:2]
        top += 15
        cv2.putText(cvorigin, cls_scr, (left, top), font, 0.5, color)

    cv2.imwrite(output_image, cvorigin)


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    segmenter = Yolov11Segmenter(
        args.model_prefix,
        args.vdsp_params,
        args.elf_file,
        batch_size,
        args.device_id,
        args.hw_config,
    )
    labels = load_labels(args.label_file)
    image_format = segmenter.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Read image failed:{args.input_file}"
        vsx_image = cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        outputs = segmenter.process(vsx_image)
        classes, scores, boxes = outputs[:3]
        det_count = outputs[4][0]
        if det_count == 0:
            print("No object detected in image.")
        else:
            for i in range(det_count):
                if scores[i] < args.threshold:
                    continue
                print(
                    f"Object class: {labels[int(classes[i])]}, score: {scores[i]}, bbox: {boxes[i].tolist()}"
                )
            if args.output_file != "":
                save_mask(
                    outputs, args.input_file, args.output_file, labels, args.threshold
                )
    else:
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = f.readlines()
        for filename in tqdm(filelist):
            filename = os.path.join(args.dataset_root, filename.replace("\n", ""))
            cv_image = cv2.imread(filename)
            assert cv_image is not None, f"Read image failed:{filename}"
            vsx_image = cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            outputs = segmenter.process(vsx_image)
            basename, _ = os.path.splitext(os.path.basename(filename))
            npz_file = os.path.join(args.dataset_output_folder, basename + ".npz")
            if len(outputs) == 5:
                det_num = outputs[4][0]
                classes = outputs[0][:det_num]
                scores = outputs[1][:det_num]
                boxes = outputs[2][:det_num, :]
                masks = outputs[3][:det_num, :, :]

                np.savez(
                    npz_file,
                    classes=classes,
                    scores=scores,
                    boxes=boxes,
                    masks=masks,
                    num=outputs[4],
                )
            else:
                print("No object detected in image.")
