
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :    zpwen
@Email  :    algorithm@vastaitech.com
@Time   :    2025/04/24 16:17:38
'''

import argparse
import glob
import json
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict,  List, Union
from multiprocessing.pool import ThreadPool
from utils import (non_max_suppression, DFL, dist2bbox, make_anchors, process_mask_upsample,
                    coco80_to_coco91_class, coco_names, xyxy2xywh, scale_image)

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
NUM_PROCESS = 5

class Segmenter:
    def __init__(
        self,
        model_size: Union[int, list],
        classes: Union[str, List[str]],
        threashold: float = 0.01
    ) -> None:
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
        self.jdict = []

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

    def scale_coords_mask(self, img1_shape, coords, img0_shape, ratio_pad=None):
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
        return int(pad[0]/gain), int(pad[1]/gain)

    def clip_coords(self, boxes, shape):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def postprocess(self, stream_ouput, classes_list, image_file, save_dir, save_img=False, **kwargs):
        origin_img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        height, width, _ = origin_img.shape
        file_name = os.path.basename(image_file)
        os.makedirs(save_dir, exist_ok=True)

        dfl = DFL(16)
        for i in range(10):
            
            stream_ouput[i] = torch.Tensor(stream_ouput[i].reshape(output_shape[i]))
        ## detect cat
        output = []
        for i in range(3):
            x = torch.cat((stream_ouput[i*2], stream_ouput[i*2+1]), 1)
            output.append(x)

        anchors, strides = (x.transpose(0, 1) for x in make_anchors(output, [8, 16, 32], 0.5))

        x_cat = torch.cat([xi.view(1, 144, -1) for xi in output], 2)
        box = x_cat[:, :16 * 4]
        cls = x_cat[:, 16 * 4:]

        dbox = dist2bbox(dfl(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
        ty = torch.cat((dbox, cls.sigmoid()), 1)
       
        det_out = (ty, output)

        ## segment cat
        tmc = []
        for i in range(6, 9):
            tmc.append(stream_ouput[i].view(1, 32, -1))
        mc = torch.cat(tmc, 2)
        p = stream_ouput[-1]

        mask_out = (torch.cat([det_out[0], mc], 1), (det_out[1], mc, p))
        pred = non_max_suppression(mask_out[0])[0]
        proto = mask_out[1][-1] if len(mask_out[1]) == 3 else mask_out[1]
        proto = proto[0]

        ### Masks
        res_length = len(pred)
        
        if res_length:
            pred_masks = process_mask_upsample(proto, pred[:, 6:], pred[:, :4], shape=self.model_size)
            pred[:, :4] = self.scale_coords(self.model_size, pred[:, :4], [height, width, 3]).round()
        else:
            return
        
        pred = pred.numpy()

        new_filename = os.path.splitext(file_name)[0]
        pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
        scale_r = self.model_size[0] / max(height, width)
        pad_h = (self.model_size[0] - height * scale_r) / 2
        pad_w = (self.model_size[0] - width * scale_r) / 2
        pred_masks = scale_image(pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                                        (height, width),
                                        ratio_pad=((scale_r, scale_r), (pad_w, pad_h)))
        
        COLORS = np.random.uniform(0, 255, size=(len(classes_list), 3))


        # draw_images
        box_list = []
        if res_length:
            for index in range(res_length):
                label = classes_list[pred[index][5].astype(np.int8)]
                score = pred[index][4]
                bbox = pred[index][:4].tolist()
                p1, p2 = (int(bbox[0]), int(bbox[1])
                          ), (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(origin_img, p1, p2, (0, 255, 255),
                              thickness=1, lineType=cv2.LINE_AA)
                text = f"{label}: {round(score * 100, 2)}%"
                y = int(int(bbox[1])) - 15 if int(int(bbox[1])
                                                  ) - 15 > 15 else int(int(bbox[1])) + 15
                cv2.putText(
                    origin_img,
                    text,
                    (int(bbox[0]), y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLORS[pred[index][5].astype(np.int8)],
                    2,
                )
                box_list.append(f"{label} {score} {int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])}")
            
            mask = np.zeros((origin_img.shape[0], origin_img.shape[1], 3), dtype=np.uint8)
            mask_bool = pred_masks.astype(np.bool_)
            for i in range(pred_masks.shape[2]):
                mask[mask_bool[:, :, i]] = COLORS[pred[i][5].astype(np.int8)]
                merged_img = cv2.addWeighted(origin_img, 0.7, mask, 0.3, 0)
            if save_img:
                cv2.imwrite(f"{save_dir}/{file_name}", merged_img)
        
        self.pred_to_json(box_list, new_filename,  pred_masks)


    def pred_to_json(self, box_list, file_name,  pred_masks):
        """Save one JSON result."""
        # Example result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        from pycocotools.mask import encode  # noqa

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict."""
            rle = encode(np.asarray(x[:, :, None], order='F', dtype='uint8'))[0]
            rle['counts'] = rle['counts'].decode('utf-8')
            return rle
        pred_masks = np.transpose(pred_masks, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)

        coco_num = coco80_to_coco91_class()
        image_id = int(file_name) if file_name.isnumeric() else file_name
        box = []
        label = []
        score = []
        for line in box_list:
            line = line.strip().split()
            label.append(coco_num[coco_names.index(" ".join(line[:-5]))])
            box.append([float(l) for l in line[-4:]])
            score.append(float(line[-5]))
        if len(box):
            box = xyxy2xywh(np.array(box))  # x1y1wh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner

            for i in range(len(box.tolist())):
                self.jdict.append({
                    "image_id": image_id,
                    "category_id": label[i],
                    "bbox": [x for x in box[i].tolist()],
                    "score": score[i],
                    'segmentation': rles[i]
                    }
                )


    def save_json(self, json_save_dir):
        with open(json_save_dir+"/predictions.json", 'w') as f:
            json.dump(self.jdict, f)  # flatten and save

    def npz_decode(self, input_image_path: str, output_npz_file: str, json_save_dir, save_img=False):
        stream_ouput = []
        for i in range(10):
            stream_ouput.append(np.load(output_npz_file, allow_pickle=True)["output_"+str(i)])

        # post proecess
        self.postprocess(stream_ouput,
                         self.classes, input_image_path, json_save_dir,  save_img)
        return stream_ouput

def npz2json(args):
    decoder = Segmenter(
        model_size=args.model_size,
        classes=args.label_txt,
        threashold=0.01
    )
    json_save_dir = args.save_dir
    os.makedirs(json_save_dir,mode=0o777, exist_ok=True)
    with open(args.datalist_txt, 'r') as f:
        input_npz_files = f.readlines()
    npz_files = glob.glob(os.path.join(args.vamp_output + "/*.npz"))
    npz_files.sort()

    for index, npz_file in enumerate(tqdm(npz_files)):
        image_path = os.path.join(args.input_image, os.path.basename(
            input_npz_files[index].strip().replace('npz', 'jpg')))
        decoder.npz_decode(image_path, npz_file, json_save_dir, args.save_image)
    decoder.save_json(json_save_dir)

### 针对size为640输入的模型
### 1024 需要修改
# output_shape = [[1, 64, 80 ,80], [1, 80, 80, 80], [1, 64, 40, 40], [1, 80, 40, 40], [1, 64, 20, 20],
#                 [1, 80, 20, 20], [1, 32, 80, 80], [1, 32, 40, 40], [1, 32, 20, 20], [1,32 ,160, 160]]
output_shape = [[1, 64, 128, 128], [1, 80, 128, 128], [1, 64, 64, 64], [1, 80, 64, 64], [1, 64, 32, 32],
                [1, 80, 32, 32], [1, 32, 128, 128], [1, 32, 64, 64], [1, 32, 32, 32], [1, 32, 256, 256]
]

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="BENCHMARK FOR OD")
    parse.add_argument("--save-dir", type=str, default="./TEMP", help="path to save result json")
    parse.add_argument(
        "--label-txt", type=str, default="coco.txt", help="label txt"
    )
    parse.add_argument(
        "--input-image",
        type=str,
        default="backbone_quant",
        help="input source image folder",
    )
    parse.add_argument(
        "--model-size",
        nargs='+',
        type=int,
        default=[1024, 1024],
    )
    parse.add_argument(
        "--datalist-txt",
        type=str,
        default="npz_datalist.txt",
        help="vamp datalist folder",
    )
    parse.add_argument(
        "--vamp-output",
        type=str,
        default="vamp_output",
        help="vamp output folder",
    )
    parse.add_argument(
        "--save-image",
        action="store_true"
    )
    args = parse.parse_args()
    print(args)

    npz2json(args)
