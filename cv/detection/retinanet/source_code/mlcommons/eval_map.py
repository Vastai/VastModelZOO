# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import glob
import json
import os

import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def get_jdict_from_txt(folder, dt_path):
    jdict = []
    # coco_num = coco80_to_coco91_class()
    files = glob.glob(os.path.join(folder, "*.txt"))
    for file_path in tqdm(files):
        with open(file_path, "r") as fout:
            data = fout.readlines()
        file_name = os.path.basename(file_path).replace('.txt', '.jpg')
        image_id = image2id[file_name]
        box = []
        label = []
        score = []
        for line in data:
            line = line.strip().split()
            label.append(class2id[" ".join(line[:-5])])
            box.append([float(l) for l in line[-4:]])
            score.append(float(line[-5]))
        if len(box) == 0:
            continue

        box = xyxy2xywh(np.array(box))  # x1y1wh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for i in range(len(box.tolist())):
            jdict.append(
                {
                    "image_id": image_id,
                    "category_id": label[i],
                    "bbox": [x for x in box[i].tolist()],
                    "score": score[i], 
                    'iscrowd': 0.0
                }
            )

    with open(dt_path, "w") as f:
        json.dump(jdict, f)


def coco_map(txt_path, gt_path, format):
    dt_path = os.path.join(txt_path, "pred.json")
    get_jdict_from_txt(txt_path, dt_path)
    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(dt_path)
    imgIds = cocoGt.getImgIds()
    print("get %d images" % len(imgIds))
    imgIds = sorted(imgIds)
    cocoEval = COCOeval(cocoGt, cocoDt, format)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # copy-paste style
    from collections import OrderedDict

    eval_results = OrderedDict()
    metric = format
    metric_items = ["mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l"]
    coco_metric_names = {
        "mAP": 0,
        "mAP_50": 1,
        "mAP_75": 2,
        "mAP_s": 3,
        "mAP_m": 4,
        "mAP_l": 5,
        "AR@100": 6,
        "AR@300": 7,
        "AR@1000": 8,
        "AR_s@1000": 9,
        "AR_m@1000": 10,
        "AR_l@1000": 11,
    }

    for metric_item in metric_items:
        key = f"{metric}_{metric_item}"
        val = float(f"{cocoEval.stats[coco_metric_names[metric_item]]:.3f}")
        eval_results[key] = val
    ap = cocoEval.stats[:6]
    eval_results[f"{metric}_mAP_copypaste"] = f"{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} {ap[4]:.3f} {ap[5]:.3f}"
    print(dict(eval_results))
    return eval_results


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="TEST BENCHMARK FOR OD")
    parse.add_argument("--format", type=str, default="bbox", help="'segm', 'bbox', 'keypoints'")
    parse.add_argument("--gt", type=str, default="openimages/annotations/openimages-mlperf.json", help="gt json")
    parse.add_argument("--txt", type=str, default="vsx_results_vacc_fp16_libtvm_1231731", help="txt files")
    args = parse.parse_args()
    print(args)

    openimages = COCO(args.gt)
    id2class = {key: value['name'] for key, value in openimages.cats.items()}
    class2id = {value: key for key, value in id2class.items()}
    id2image = {key: value['file_name'] for key, value in openimages.imgs.items()}
    image2id = {value: key for key, value in id2image.items()}
    coco_map(txt_path=args.txt, gt_path=args.gt, format=args.format)


'''
[1, 3, 800, 800]
onnx
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.534
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.418
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.037
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.131
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.426
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.603
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.629
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.096
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.678
{'bbox_mAP': 0.385, 'bbox_mAP_50': 0.534, 'bbox_mAP_75': 0.418, 'bbox_mAP_s': 0.037, 'bbox_mAP_m': 0.131, 'bbox_mAP_l': 0.427, 'bbox_mAP_copypaste': '0.385 0.534 0.418 0.037 0.131 0.427'}


bench  99%*0.375 = 0.37125

vsx_results_vacc_fp16_libtvm_1231731
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.534
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.416
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.426
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.603
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.629
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.104
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.334
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.678
{'bbox_mAP': 0.385, 'bbox_mAP_50': 0.534, 'bbox_mAP_75': 0.416, 'bbox_mAP_s': 0.039, 'bbox_mAP_m': 0.128, 'bbox_mAP_l': 0.426, 'bbox_mAP_copypaste': '0.385 0.534 0.416 0.039 0.128 0.426'}

vacc_int8-kl_divergence BILINEAR_PILLOW
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.384
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.531
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.416
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.037
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.425
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.602
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.101
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.331
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.677
{'bbox_mAP': 0.384, 'bbox_mAP_50': 0.531, 'bbox_mAP_75': 0.416, 'bbox_mAP_s': 0.037, 'bbox_mAP_m': 0.128, 'bbox_mAP_l': 0.425, 'bbox_mAP_copypaste': '0.384 0.531 0.416 0.037 0.128 0.425'}
'''