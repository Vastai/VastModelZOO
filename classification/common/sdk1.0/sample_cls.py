'''
Copyright (C) 2022-2023 VASTAI Technologies Co., Ltd. All Rights Reserved.

The information contained herein is confidential property of the company.
The user, copying, transfer or disclosure of such information is prohibited
except by express written agreement with VASTAI Technologies Co., Ltd.
'''
import argparse
import glob
import os
import time

from image_classification import Classifier
from tqdm import tqdm

parse = argparse.ArgumentParser(description="RUN CLS WITH VACL")
parse.add_argument("--task", type=str, default="topk", choices=["run", "topk"])
parse.add_argument("--file_path", type=str, default="/path/to/data/eval/ILSVRC2012_img_val", help="img or dir  path")
parse.add_argument("--model_info", type=str, default="/path/to/vastmodelzoo/classification/common/sdk1.0/model_info/model_info_model.json", help="model info")
parse.add_argument("--vdsp_params_info", type=str, default="/path/to/vastmodelzoo/classification/common/sdk1.0/model_info/vdsp_params_model_rgb.json", help="vdsp op info")
parse.add_argument("--label_txt", type=str, default="/path/to/data/eval/imagenet.txt", help="label txt")
parse.add_argument("--topk", type=int, default=5, help="top1 or top5")
parse.add_argument("--device_id", type=int, default=0, help="device id")
parse.add_argument("--batch", type=int,default=1,help="bacth size")
parse.add_argument("--save_dir", type=str, default="/path/to/weights/sdk", help="save_dir")

args = parse.parse_args()
print(args)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


classifier = Classifier(
    model_info=args.model_info,
    vdsp_params_info=args.vdsp_params_info,
    classes=args.label_txt,
    topk  = args.topk,
    device_id=args.device_id,
    batch_size=args.batch,
)

# 测试
if args.task == "run":
    result = classifier.classify(args.file_path)
    print(f"{args.file_path} => {result}")
    cls_id = result[0]
    label_list = result[1]
    score_list = result[2]
    for k in range(args.topk):
        print(
                "{}: Relay top-{} id: {}, prob: {:.8f}, class name: {}".format(
                    args.file_path, k, cls_id[k], score_list[k], label_list[k]
                )
            )

# benchmark
else:
    fin = open(os.path.join(args.save_dir, "vacl_cls_result") + ".txt", "w")
    images = glob.glob( os.path.join(args.file_path + "/**/*"))
    time_begin = time.time()
    results = classifier.classify_batch(images)
    for (image, result) in tqdm(zip(images, results)):
        cls_id = result[0]
        label_list = result[1]
        score_list = result[2]
        for k in range(args.topk):
            fin.write(
                image
                + ": "
                + "Relay top-{} id: {}, prob: {:.8f}, class name: {}".format(
                    k, cls_id[k], score_list[k], label_list[k]
                )
                + "\n"
            )
    time_end = time.time()
    print(
        f"\n{len(images)} images in {time_end - time_begin} seconds, ({len(images) / (time_end - time_begin)} images/second)\n"
    )
    fin.close()



