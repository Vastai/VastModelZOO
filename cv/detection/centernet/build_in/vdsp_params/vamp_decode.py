# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import  os
import sys
import cv2
import glob
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../..')

from source_code.CenterNet.src.lib.utils.image import get_affine_transform
from source_code.CenterNet.src.lib.models.decode import ctdet_decode
from source_code.CenterNet.src.lib.utils.post_process import ctdet_post_process
from source_code.CenterNet.src.lib.external.nms import soft_nms

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']


def preprocess(img_file, scale=1.0):
    img_data = cv2.imread(img_file)
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

    height, width = img_data.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)
    if True:
      inp_height, inp_width = 512, 512
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | 31) + 1
      inp_width = (new_width | 31) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(img_data, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    # inp_image = cv2.resize(image, (inp_width, inp_height))

    # inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1)# .reshape(1, 3, inp_height, inp_width)
    # images = torch.from_numpy(images)

    return images, c, s


def postprocess(dets, c, s):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [c], [s],
        128, 128, 80)
    for j in range(1, 80 + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= 1.0
    return dets[0]

def merge_outputs(detections):
    results = {}
    for j in range(1, 80 + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if True:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, 80 + 1)])
    if len(scores) > 100:
      kth = len(scores) - 100
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, 80 + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

def save_result(image_path, result, save_dir, draw=False):
    os.makedirs(save_dir, exist_ok=True)
    
    origin_img = cv2.imread(image_path)

    COLORS = np.random.uniform(0, 255, size=(200, 3))
    
    save_path = os.path.join(save_dir, os.path.basename(image_path).replace(".jpg", ".txt"))
    
    with open(save_path, 'w') as ff:
        for k, v in result.items():
            for box in v:
                '''if box[-1] > 0.5:
                    output.append(box)'''
                cls = class_names[k-1]
                bb = [cls, box[-1], box[0], box[1], box[2], box[3]]
                bb = ' '.join([str(b) for b in bb])
                ff.writelines(bb + '\n')
                
                if draw:
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    cv2.rectangle(origin_img, p1, p2, COLORS[k-1], thickness=1, lineType=cv2.LINE_AA)
                    text = f"{cls}: {round(box[-1] * 100, 2)}%"
                    y = int(int(box[1])) - 15 if int(int(box[1])) - 15 > 15 else int(int(box[1])) + 15
                    cv2.putText(
                        origin_img,
                        text,
                        (int(box[0]), y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[k-1],
                        1,
                    )
        if draw:
            file_name = os.path.split(image_path)[-1]
            cv2.imwrite(os.path.join(save_dir, file_name), origin_img)


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="MAKE DATA LIST")
    parse.add_argument("--gt_dir", type=str, default="./code/eval/coco_val2017", help="path to input source image folder")
    parse.add_argument("--input_npz_path", type=str, default="./code/eval/npz_datalist_coco_eval.txt", help="vamp data list text file path")
    parse.add_argument("--out_npz_dir", type=str, default="./code/vamc/vamp/0.2.0/outputs/centernet", help="path to vamp output npz image folder")
    parse.add_argument("--draw_dir", type=str, default="coco_val2017_npz_result", help="path to result image dolder")
    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()

    psnr_list = []
    ssim_list = []
    with open(args.input_npz_path, 'r') as f:
        input_images = f.readlines()
        for index, line in enumerate(tqdm(input_images)):
            image_name = os.path.basename(line.strip('\n')).replace(".npz", ".jpg")
            image_path = os.path.join(args.gt_dir, image_name)

            # load out from vamp npz
            yolo1_layer = np.load(output_npz_list[index], allow_pickle=True)["output_0"]
            yolo2_layer = np.load(output_npz_list[index], allow_pickle=True)["output_1"]
            yolo3_layer = np.load(output_npz_list[index], allow_pickle=True)["output_2"]

            yolo1_layer = torch.Tensor(yolo1_layer)
            yolo2_layer = torch.Tensor(yolo2_layer)
            yolo3_layer = torch.Tensor(yolo3_layer)
            
            hm = yolo1_layer.sigmoid_()
            wh = yolo2_layer
            reg = yolo3_layer

            detections = []

            _, c, s = preprocess(image_path)

            dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=False, K=100)
            dets = postprocess(dets, c, s)
            detections.append(dets)
            detections = merge_outputs(detections)

            # save text and draw boxes
            save_result(image_path, detections, args.draw_dir, draw=True)


"""
deploy_weights/centernet_res18-int8-kl_divergence-3_512_512-vacc/centernet_res18
{'bbox_mAP': 0.214, 'bbox_mAP_50': 0.367, 'bbox_mAP_75': 0.218, 'bbox_mAP_s': 0.06, 'bbox_mAP_m': 0.22, 'bbox_mAP_l': 0.369, 'bbox_mAP_copypaste': '0.214 0.367 0.218 0.060 0.220 0.369'}
"""