import argparse
import glob
import json
import os
import cv2
import torch
import shutil
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm
from typing import Dict, Generator, Iterable, List, Union

import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../..')
from source_code.utils import AnchorGenerator, yolo_data_process_cv, delta2bbox, non_max_suppression, scale_coords

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

    def postprocess(self, layer0, layer1, layer2, classes_list, image_file, save_dir, save_img=False, **kwargs):
        origin_img = cv2.imread(image_file, cv2.IMREAD_COLOR)

        file_name = os.path.basename(image_file)
        os.makedirs(save_dir, exist_ok=True)

        cls_score = torch.Tensor(layer0)
        bbox_reg = torch.Tensor(layer1)
        objectness = torch.Tensor(layer2)

        anchor = AnchorGenerator([32], [1.], [1., 2, 4, 8, 16])
        all_anchors = anchor.grid_priors([(int(self.model_size[0]/32), int(self.model_size[1]/32))], device='cpu')

        target_means = (0., 0., 0., 0.),
        target_stds = (1., 1., 1., 1.),

        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, 80, H, W)

        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(
            1. + torch.clamp(cls_score.exp(), max=1e8) +
            torch.clamp(objectness.exp(), max=1e8))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)

        cls_score_torch = normalized_cls_score[0].permute(
            1, 2, 0).reshape(-1, 80)
        bbox_pred_torch = bbox_reg[0].permute(1, 2, 0).reshape(-1, 4)
        cls_score_torch = cls_score_torch.sigmoid()

        pred = delta2bbox(
            all_anchors[0], bbox_pred_torch, target_means, target_stds)

        out = torch.cat((pred, torch.ones((5*int(self.model_size[0]/32)*int(self.model_size[1]/32), 1)), cls_score_torch), axis=1)
        out = out.unsqueeze(0)
        out = non_max_suppression(out, 0.05, 0.6)[0]
        if len(out):
            out[:, :4] = scale_coords(
                [self.model_size[0], self.model_size[1]], out[:, :4], origin_img.shape).round()

        res_length = len(out)
        COLORS = np.random.uniform(0, 255, size=(len(classes_list), 3))
        fin = open(f"{save_dir}/{os.path.splitext(file_name)[0]}.txt", "w")
        # 画框
        if res_length:
            for index in range(res_length):
                label = classes_list[int(out[index, 5])]
                score = float(out[index, 4])
                bbox = out[index, :4].tolist()
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
                    COLORS[int(out[index, 5])],
                    2,
                )
                fin.write(
                    f"{label} {score} {int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])}\n")
            if save_img:
                cv2.imwrite(f"{save_dir}/{file_name}", origin_img)
        fin.close()

    def npz_decode(self, input_image_path: str, output_npz_file: str, txt_save_dir):
        # print(output_npz_file)
        #print(np.load(output_npz_file, allow_pickle=True).files)
        layer0 = np.load(output_npz_file, allow_pickle=True)["output_0"]
        layer1 = np.load(output_npz_file, allow_pickle=True)["output_1"]
        layer2 = np.load(output_npz_file, allow_pickle=True)["output_2"]
        stream_ouput = (layer0, layer1, layer2)

        # post proecess
        self.postprocess(layer0, layer1, layer2,
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
        default=[1280,1280],
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

'''
fp16
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.570
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.194
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.425
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.610
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.694
'''