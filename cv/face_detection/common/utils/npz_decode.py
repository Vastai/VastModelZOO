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
import shutil
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm
from typing import Dict, Generator, Iterable, List, Union

class Decoder:
    def __init__(
        self,
        model_size: Union[int, list],
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
        self.threashold = threashold

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            # gain  = old / new
            gain = min(img1_shape[0] / img0_shape[0],
                       img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / \
                2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain

        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self, boxes, shape):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def postprocess(self, output_cls, output_box, output_lmd, img_dir, image_file, save_dir, save_img=False, **kwargs):
        origin_img = cv2.imread(image_file, cv2.IMREAD_COLOR)

        file_name = os.path.basename(image_file)
        os.makedirs(save_dir, exist_ok=True)
        
        im_height, im_width, _ = origin_img.shape
        # hscale = im_height / self.model_size[0]
        # wscale = im_width / self.model_size[0]

        # post processing
        r = min(self.model_size[0] / im_width, self.model_size[1] / im_height)
        unpad_w = int(round(im_width * r))
        unpad_h = int(round(im_height * r))
        dw = self.model_size[0] - unpad_w
        dh = self.model_size[1] - unpad_h
        dw /= 2
        dh /= 2

        dirname = os.path.join(save_dir, img_dir)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        result_name = os.path.join(save_dir, img_dir, file_name.replace('jpg', 'txt'))
        
        with open(result_name, "w") as fd:
            fd.write(file_name + '\n')
            box_num = 0
            for index in range(output_cls[0].shape[0]):
                if output_cls[0][index] == -1:
                    break
                box_num += 1
            fd.write(str(box_num) + '\n')
            for index in range(output_cls[0].shape[0]):
                if output_cls[0][index] == -1:
                    break
                box = output_box[0][index]
                confidence = output_cls[0][index][0]

                # 左上角点，w,h
                # x = int(box[0] * wscale)
                # y = int(box[1] * hscale)
                # w = int(box[2] * wscale) - int(box[0] * wscale)
                # h = int(box[3] * hscale) - int(box[1] * hscale)

                x = (box[0] - dw) * im_width / unpad_w
                x2 = (box[2] - dw) * im_width / unpad_w
                y = (box[1] - dh) * im_height / unpad_h
                y2 = (box[3] - dh) * im_height / unpad_h
                w = x2 - x
                h = y2 - y
                
                confidence = str(confidence)
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

    def npz_decode(self, img_dir, input_image_path: str, output_npz_file: str, txt_save_dir):
        # print(output_npz_file)
        #print(np.load(output_npz_file, allow_pickle=True).files)
        output_cls = np.load(output_npz_file, allow_pickle=True)["output_0"]
        output_box = np.load(output_npz_file, allow_pickle=True)["output_1"]
        output_lmd = np.load(output_npz_file, allow_pickle=True)["output_2"]
        stream_ouput = (output_cls, output_box, output_lmd)

        # post proecess
        self.postprocess(output_cls, output_box, output_lmd, img_dir, input_image_path, txt_save_dir,  save_img=False)

        return stream_ouput

def npz2txt(args):
    decoder = Decoder(
        model_size=args.model_size,
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
    img_dir_list = os.listdir(args.input_image_dir)
    img_dir = {}
    for i in img_dir_list:
        num = i.split('--')[0]
        img_dir[num] = i

    for index, npz_file in enumerate(tqdm(npz_files)):

        img_dir = os.listdir(args.input_image_dir)

        for dir in img_dir:
            
            ## 不限vamp_input_list后缀
        
            image_path = os.path.join(args.input_image_dir, dir, os.path.basename(
                input_npz_files[index].strip().replace('npz', 'jpg')))
            # print(image_path)
            # print(os.path.exists(image_path))
            if os.path.exists(image_path):
                result = decoder.npz_decode(dir, image_path, npz_file, txt_save_dir)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="TEST BENCHMARK FOR OD")
    parse.add_argument("--format", type=str, default="bbox", help="'segm', 'bbox', 'keypoints'")
    parse.add_argument("--txt", type=str, default="./TEMP_TXT", help="txt files")
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
