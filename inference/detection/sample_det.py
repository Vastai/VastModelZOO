import argparse
import glob
import os
import time

import cv2 as cv
import numpy as np
from image_detection import Detector

parse = argparse.ArgumentParser(description="RUN Det WITH VACL")
parse.add_argument(
    "--file_path",
    type=str,
    default="/workspace/VastDeploy_backup/data/eval/coco_val2017",
    help="img or dir  path",
)
parse.add_argument("--model_info", type=str, default="./info/model_info_yolov3.json", help="model info")
parse.add_argument(
    "--vdsp_params_info",
    type=str,
    default="./info/vdsp_params_yolov3_letterbox_rgb.json",
    help="vdsp op info",
)
parse.add_argument(
    "--label_txt", type=str, default="/workspace/vastdeploy/data/label/coco.txt", help="label txt"
)
parse.add_argument("--device_id", type=int, default=0, help="device id")
parse.add_argument("--batch", type=int, default=1, help="bacth size")
parse.add_argument("--save_dir", type=str, default="../../output", help="save_dir")
args = parse.parse_args()


def save_result(image_path, result, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    COLORS = np.random.uniform(0, 255, size=(200, 3))
    fin = open(os.path.join(save_dir, os.path.splitext(os.path.split(image_path)[-1])[0] + ".txt"), "w")
    origin_img = cv.imread(image_path)
    for out_index in range(len(result[0])):
        p1, p2 = (int(result[2][out_index][0]), int(result[2][out_index][1])), (
            int(result[2][out_index][2]) + int(result[2][out_index][0]),
            int(result[2][out_index][3]) + int(result[2][out_index][1]),
        )
        cv.rectangle(origin_img, p1, p2, COLORS[int(result[0][out_index])], thickness=1, lineType=cv.LINE_AA)
        text = f"{result[3][out_index]}: {round(result[1][out_index] * 100, 2)}%"
        y = int(result[2][out_index][1]) - 15 if int(result[2][out_index][1]) - 15 > 15 else int(result[2][out_index][1]) + 15
        cv.putText(
            origin_img,
            text,
            (int(result[2][out_index][0]), y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLORS[int(result[0][out_index])],
            1,
        )
        fin.write(
            f"{result[3][out_index]} {result[1][out_index]} {result[2][out_index][0]} {result[2][out_index][1]} {result[2][out_index][2]+result[2][out_index][0]} {result[2][out_index][3]+ result[2][out_index][1]}\n"
        )
    file_name = os.path.split(image_path)[-1]
    cv.imwrite(os.path.join(save_dir, file_name), origin_img)
    fin.close()


detector = Detector(
    model_info=args.model_info,
    vdsp_params_info=args.vdsp_params_info,
    classes=args.label_txt,
    device_id=args.device_id,
    batch_size=args.batch,
)

if os.path.isfile(args.file_path):
    result = detector.detection(args.file_path)
    print(f"{args.file_path} => {result}")
    save_result(args.file_path, result, args.save_dir)

else:
    # Test multiple images
    images = glob.glob(os.path.join(args.file_path + "/*"))
    time_begin = time.time()
    results = detector.detection_batch(images)
    for (image, result) in zip(images, results):
        print(f"{image} => {result}")
        save_result(image, result, args.save_dir)
    time_end = time.time()

    print(
        f"\n{len(images)} images in {time_end - time_begin} seconds, ({len(images) / (time_end - time_begin)} images/second)\n"
    )
