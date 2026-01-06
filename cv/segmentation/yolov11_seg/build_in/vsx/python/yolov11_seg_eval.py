# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import os
import sys
import glob
import json
import argparse
import tqdm
from pycocotools.mask import encode  # noqa
from pycocotools.coco import COCO  # noqa
from pycocotools.cocoeval import COCOeval  # noqa


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    x = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]
    return x


def single_encode(x):
    """Encode predicted masks as RLE and append results to jdict."""
    rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EVAL YOLOV11_SEG")
    parser.add_argument(
        "--output_path",
        default="/path/to/infer/output/path",
    )
    parser.add_argument("--gt", type=str, default="/path/to/instances_val2017.json")
    args = parser.parse_args()

    coco_num = coco80_to_coco91_class()

    output_path = args.output_path
    file_list = glob.glob(
        os.path.join(
            output_path,
            "*npz",
        )
    )
    jdict = []
    for _, file in enumerate(tqdm.tqdm(file_list, file=sys.stdout)):
        tensors = []
        npz_data = np.load(file.strip())
        image_id = os.path.splitext(os.path.basename(file))[0]
        image_id = int(image_id)

        classes = npz_data["classes"]
        scores = npz_data["scores"]
        boxes = npz_data["boxes"]
        masks = npz_data["masks"]
        num = npz_data["num"][0]

        for i in range(num):
            box = boxes[i].tolist()
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]

            jdict.append(
                {
                    "image_id": image_id,
                    "category_id": coco_num[int(classes[i])],
                    "bbox": [x for x in box],
                    "score": scores[i].astype(float),
                    "segmentation": single_encode(masks[i]),
                }
            )

    with open("yolov11seg_predictions.json", "w") as f:
        json.dump(jdict, f)  # flatten and save

    try:
        anno = COCO(str(args.gt))  # init annotations api
        pred = anno.loadRes(
            str("yolov11seg_predictions.json")
        )  # init predictions api (must pass string, not Path)
        imgIds = anno.getImgIds()
        print("get %d images" % len(imgIds))
        imgIds = sorted(imgIds)
        for i, eval in enumerate(
            [COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm")]
        ):
            eval.params.imgIds = imgIds
            eval.evaluate()
            eval.accumulate()
            eval.summarize()

    except Exception as e:
        print(f"pycocotools unable to run: {e}")
