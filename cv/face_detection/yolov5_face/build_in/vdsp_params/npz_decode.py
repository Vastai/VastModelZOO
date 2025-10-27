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
import cv2
import torch
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, Generator, Iterable, List, Union

import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../..')

from source_code.utils.general import non_max_suppression_face, scale_coords, xyxy2xywh

class Decoder:
    def __init__(
        self,
        model_size: Union[int, list],
        classes: Union[str, List[str]],
        threashold: float = 0.01
    ) -> None:
        # if isinstance(vdsp_params_path, str):
        #     with open(vdsp_params_path) as f:
        #         vdsp_params_dict = json.load(f)


        if isinstance(model_size,int):
            self.model_size = [model_size, model_size]
        else:
            # h,w
            assert len(model_size) == 2 , "model_size ERROR."
            self.model_size = model_size # h,w
        self.classes = ['face']
        self.threashold = threashold

    def postprocess(self, out0, out1, out2, classes_list, image_file, save_dir, save_img=False, **kwargs):
        origin_img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        h, w, c = origin_img.shape

        file_name = os.path.basename(image_file)
        os.makedirs(save_dir, exist_ok=True)

        grid = [torch.zeros(1)] * 3
        stride = [8, 16, 32]
        anchor_grid = [[4,5,  8,10,  13,16], [23,29,  43,55,  73,105], [146,217,  231,300,  335,433]]
        anchor_grid = torch.Tensor(anchor_grid).clone().view(3, 1, -1, 1, 1, 2)
        z = []

        out = [out0, out1, out2]


        for i in range(3):
            yolo_layer = torch.Tensor(out[i])
            bs, _, ny, nx = yolo_layer.shape
            yolo_layer = yolo_layer.view(1, 3, 16, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            y = torch.full_like(yolo_layer, 0)
            class_range = list(range(5)) + list(range(15,15+1))
            y[..., class_range] = yolo_layer[..., class_range].sigmoid()
            y[..., 5:15] = yolo_layer[..., 5:15]

            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid[i] = torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()

            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh

            y[..., 5:7]   = y[..., 5:7] *   anchor_grid[i] + grid[i] * stride[i] # landmark x1 y1
            y[..., 7:9]   = y[..., 7:9] *   anchor_grid[i] + grid[i] * stride[i]# landmark x2 y2
            y[..., 9:11]  = y[..., 9:11] *  anchor_grid[i] + grid[i] * stride[i]# landmark x3 y3
            y[..., 11:13] = y[..., 11:13] * anchor_grid[i] + grid[i] * stride[i]# landmark x4 y4
            y[..., 13:15] = y[..., 13:15] * anchor_grid[i] + grid[i] * stride[i]# landmark x5 y5

            z.append(y.view(bs, -1, 16))

        o = torch.cat(z, 1)
        pred = non_max_suppression_face(o, 0.02, 0.5)

        COLORS = np.random.uniform(0, 255, size=(len(classes_list), 3))
        if not os.path.isdir(os.path.join(save_dir, image_file.split('/')[-2])):
            os.makedirs(os.path.join(save_dir, image_file.split('/')[-2]))
        fin = open(os.path.join(save_dir, image_file.split('/')[-2], image_file.split('/')[-1].replace('jpg', 'txt')), 'w')
        file_name = os.path.basename(image_file)[:-4]
        fin.writelines(file_name + '\n')
        fin.write(str(pred[0].shape[0]) + '\n')

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords([640, 640], det[:, :4], (h, w, 3))#.round()

                for d in det:

                    box = [float(x) for x in d[:4]]
                    box[2] = box[2] - box[0]
                    box[3] = box[3] - box[1]

                    conf = d[4].cpu().numpy()

                    box = [box[0], box[1], box[2], box[3], float(conf if conf <= 1 else 1)]
                    save_txt = ' '.join([str(bb) for bb in box])
                    fin.writelines(save_txt + '\n')
        fin.close()

    def npz_decode(self, input_image_path: str, output_npz_file: str, txt_save_dir):
        # print(output_npz_file)
        #print(np.load(output_npz_file, allow_pickle=True).files)
        out0 = np.load(output_npz_file, allow_pickle=True)["output_0"]
        out1 = np.load(output_npz_file, allow_pickle=True)["output_1"]
        out2 = np.load(output_npz_file, allow_pickle=True)["output_2"]
        stream_ouput = (out0, out1, out2)

        # post proecess
        self.postprocess(out0, out1, out2,
                         self.classes, input_image_path, txt_save_dir,  save_img=False)

        return stream_ouput

def npz2txt(args):
    decoder = Decoder(
        model_size=args.model_size,
        classes=args.label_txt,
        threashold=0.01
    )

    txt_save_dir = args.txt
    if os.path.exists(txt_save_dir):
        shutil.rmtree(txt_save_dir)
    os.makedirs(txt_save_dir,mode=0o777, exist_ok=True)

    with open(args.vamp_datalist_path, 'r') as f:
        input_npz_files = f.readlines()

    npz_files = glob.glob(os.path.join(args.vamp_output_dir + "/*.npz"))
    npz_files.sort()

    for index, npz_file in enumerate(tqdm(npz_files)):

        img_dir = os.listdir(args.input_image_dir)

        for dir in img_dir:

            ## 不限vamp_input_list后缀
            image_path = os.path.join(args.input_image_dir, dir, os.path.basename(
                input_npz_files[index].strip().replace('npz', 'jpg')))
            if os.path.exists(image_path):
                # print(image_path)
                # print(os.path.exists(image_path))
                result = decoder.npz_decode(image_path, npz_file, txt_save_dir)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="TEST BENCHMARK FOR OD")
    parse.add_argument("--format", type=str, default="bbox", help="'segm', 'bbox', 'keypoints'")
    parse.add_argument("--txt", type=str, default="./TEMP_TXT", help="txt files")
    parse.add_argument(
        "--label_txt", type=str, default=None, help="label txt"
    )
    parse.add_argument(
        "--input_image_dir",
        type=str,
        default="./source_data/dataset/val2017",
        help="input source image folder",
    )
    parse.add_argument(
        "--model_size",
        nargs='+',
        type=int,
        default=[640,640],
    )
    parse.add_argument(
        "--vamp_datalist_path",
        type=str,
        default="./outputs/data_npz_datalist.txt",
        help="vamp datalist folder",
    )
    parse.add_argument(
        "--vamp_output_dir",
        type=str,
        default="./outputs/model_latency_npz",
        help="vamp output folder",
    )
    args = parse.parse_args()
    print(args)

    npz2txt(args)
