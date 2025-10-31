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
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm
from typing import Dict, Generator, Iterable, List, Union

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
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    # clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

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

        if isinstance(classes, str):
            with open(classes) as f:
                classes = [cls.strip() for cls in f.readlines()]

        if isinstance(model_size,int):
            self.model_size = [model_size, model_size]
        else:
            # h,w
            assert len(model_size) == 2 , "model_size ERROR."
            self.model_size = model_size # h,w
        self.classes = classes
        self.threashold = threashold

    def postprocess(self, out, classes_list, image_file, save_dir, save_img=False, **kwargs):
        origin_img = cv2.imread(image_file, cv2.IMREAD_COLOR)

        file_name = os.path.basename(image_file)
        os.makedirs(save_dir, exist_ok=True)

        output1 = torch.Tensor(out[0])
        output2 = torch.Tensor(out[1])
        output3 = torch.Tensor(out[2])
        output4 = torch.Tensor(out[3])

        import sys
        _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
        sys.path.append(_cur_file_path + os.sep + '../..')
        from source_code.nanodet.nanodet_plus_head import NanoDetPlusHead
        from source_code.nanodet.transform import Pipeline
        
        pipeline = Pipeline({"normalize": [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]}, False)
        head = NanoDetPlusHead(80, None, 3, strides=[8, 16, 32, 64])
        outputs = [torch.Tensor(output1).flatten(start_dim=2), torch.Tensor(output2).flatten(start_dim=2), torch.Tensor(output3).flatten(start_dim=2), torch.Tensor(output4).flatten(start_dim=2)]
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        img_info = {"id": 0}
        img_info["file_name"] = None
        img_info["height"] = [416]
        img_info["width"] = [416]
        img, _, _ = letterbox(origin_img, (416, 416))
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = pipeline(None, meta, [416, 416])
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).unsqueeze(0)
        result = head.post_process(outputs, meta)[0]
        # result_img = head.show_result(meta["raw_img"], result, classes_list, score_thres=0.35, show=False)
        # cv2.imwrite('test.jpg', result_img)
        fin = open(f"{save_dir}/{os.path.splitext(file_name)[0]}.txt", "w")

        for id, bbox in result.items():
            if len(bbox) != 0:
                for _box in bbox:
                    b = np.array(_box[:4]).reshape((1, -1))
                    b = scale_coords((416, 416), b, origin_img.shape).round()[0]
                    bb = [classes_list[id], float(_box[4])] + b.tolist()
                    bb = ' '.join([str(b) for b in bb])
                    fin.writelines(bb + '\n')
        fin.close()

    def npz_decode(self, input_image_path: str, output_npz_file: str, txt_save_dir):
        # print(output_npz_file)
        #print(np.load(output_npz_file, allow_pickle=True).files)
        layer0 = np.load(output_npz_file, allow_pickle=True)["output_0"]
        layer1 = np.load(output_npz_file, allow_pickle=True)["output_1"]
        layer2 = np.load(output_npz_file, allow_pickle=True)["output_2"]
        layer3 = np.load(output_npz_file, allow_pickle=True)["output_3"]
        stream_ouput = (layer0, layer1, layer2, layer3)

        # post proecess
        self.postprocess(stream_ouput, self.classes, input_image_path, txt_save_dir,  save_img=False)

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

        ## 不限vamp_input_list后缀
        image_path = os.path.join(args.input_image_dir, os.path.basename(
            input_npz_files[index].strip().replace('npz', 'jpg')))
        # print(image_path)
        # print(os.path.exists(image_path))
        result = decoder.npz_decode(image_path, npz_file, txt_save_dir)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="TEST BENCHMARK FOR OD")
    parse.add_argument("--format", type=str, default="bbox", help="'segm', 'bbox', 'keypoints'")
    parse.add_argument("--txt", type=str, default="./TEMP_TXT", help="txt files")
    parse.add_argument(
        "--label_txt", type=str, default="./configs/coco.txt", help="label txt"
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
