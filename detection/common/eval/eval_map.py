import argparse
import glob
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_names = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]  # class names

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    x = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90,
    ]
    return x


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
    coco_num = coco80_to_coco91_class()
    files = glob.glob(os.path.join(folder, "*.txt"))
    for file_path in files:
        with open(file_path, "r") as fout:
            data = fout.readlines()
        file_name = os.path.splitext(os.path.split(file_path)[-1])[0]
        image_id = int(file_name) if file_name.isnumeric() else file_name
        box = []
        label = []
        score = []
        for line in data:
            line = line.strip().split()
            label.append(coco_num[coco_names.index(" ".join(line[:-5]))])
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
    parse.add_argument("--gt", type=str, default="../../data/eval/instances_val2017.json", help="gt json")
    parse.add_argument("--txt", type=str, default="../../output", help="txt files")
    args = parse.parse_args()
    print(args)

    coco_map(txt_path=args.txt, gt_path=args.gt, format=args.format)
