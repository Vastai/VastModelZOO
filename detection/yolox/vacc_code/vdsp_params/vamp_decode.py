import os
import cv2
import glob
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, Generator, Iterable, List, Union

class Decoder:
    def __init__(
        self,
        vdsp_params_path: str,
        classes: Union[str, List[str]],
        threashold: float = 0.01
    ) -> None:
        if isinstance(vdsp_params_path, str):
            with open(vdsp_params_path) as f:
                vdsp_params_dict = json.load(f)

        if isinstance(classes, str):
            with open(classes) as f:
                classes = [cls.strip() for cls in f.readlines()]

        self.model_size = [vdsp_params_dict["config"][3]["Value"], vdsp_params_dict["config"][3]["Value"]] # h,w
        self.classes = classes
        self.threashold = threashold

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
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

        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self, boxes, shape):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def postprocess(self, box_ids_np, scores_np, bboxes_np, classes_list, image_file, save_dir, draw_image=False, **kwargs):
        origin_img = cv2.imread(image_file, cv2.IMREAD_COLOR)

        file_name = os.path.basename(image_file)
        os.makedirs(save_dir, exist_ok=True)

        box_ids_np = np.squeeze(box_ids_np, axis=0)
        box_ids_np = box_ids_np[[not np.all(box_ids_np[i] == -1) for i in range(box_ids_np.shape[0])], :]
        scores_np = np.squeeze(scores_np, axis=0)
        scores_np = scores_np[[not np.all(scores_np[i] == -1) for i in range(scores_np.shape[0])], :]
        bboxes_np = np.squeeze(bboxes_np, axis=0)
        bboxes_np = bboxes_np[[not np.all(bboxes_np[i] == -1) for i in range(bboxes_np.shape[0])], :]
        # 反转尺寸
        bboxes_np = self.scale_coords(self.model_size, bboxes_np, origin_img.shape).round()
        res_length = len(bboxes_np)
        COLORS = np.random.uniform(0, 255, size=(len(classes_list), 3))
        fin = open(f"{save_dir}/{os.path.splitext(file_name)[0]}.txt", "w")
        # 画框
        if res_length:
            for index in range(res_length):
                label = classes_list[box_ids_np[index, :][0].astype(np.int8)]
                score = scores_np[index, :][0]
                bbox = bboxes_np[index, :].tolist()
                fin.write(f"{label} {score} {int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])}\n")
                if draw_image:
                    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(origin_img, p1, p2, (0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    text = f"{label}: {round(score * 100, 2)}%"
                    y = int(int(bbox[1])) - 15 if int(int(bbox[1])) - 15 > 15 else int(int(bbox[1])) + 15
                    cv2.putText(
                        origin_img,
                        text,
                        (int(bbox[0]), y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[box_ids_np[index, :][0].astype(np.int8)],
                        2,
                    )

            if draw_image:
                cv2.imwrite(f"{save_dir}/{file_name}", origin_img)
        fin.close()

    def npz_decode(self, input_image_path:str, output_npz_file:str, save_dir:str, draw_image:bool=True):

        box_ids = np.load(output_npz_file, allow_pickle=True)["output_0"]
        box_scores = np.load(output_npz_file, allow_pickle=True)["output_1"]
        box_coords = np.load(output_npz_file, allow_pickle=True)["output_2"]
        stream_ouput = (box_ids, box_scores, box_coords)

        # post proecess
        self.postprocess(box_ids, box_scores, box_coords, self.classes, input_image_path, save_dir, draw_image)

        return stream_ouput


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="vamp out decoding")
    parse.add_argument(
        "--input_image_dir",
        type=str,
        default="./eval/coco_val2017",
        help="input source image folder",
    )
    parse.add_argument(
        "--vamp_datalist_path",
        type=str,
        default="npz_datalist.txt",
        help="vamp datalist folder",
    )
    parse.add_argument(
        "--vamp_output_dir",
        type=str,
        default="./vamp/0.3.0/save/yolov5s",
        help="vamp output folder",
    )
    parse.add_argument(
        "--vdsp_params_path",
        type=str,
        default="detection/yolov5/vacc_code/vamp_info/ultralytics-yolov5s-vdsp_params.json",
        help="vdsp_params file path",
    )
    parse.add_argument(
        "--label_txt", type=str, default="./eval/coco.txt", help="label txt"
    )
    parse.add_argument(
        "--draw_image", type=bool, default=True, help="save the draw image"
    )
    parse.add_argument("--save_dir", type=str, default="output", help="save_dir")
    args = parse.parse_args()

    decoder = Decoder(
        vdsp_params_path=args.vdsp_params_path,
        classes=args.label_txt,
        threashold=0.01
    )

    with open(args.vamp_datalist_path, 'r') as f:
        input_npz_files = f.readlines()

    npz_files = glob.glob(os.path.join(args.vamp_output_dir + "/*.npz"))
    npz_files.sort()

    for index, npz_file in enumerate(tqdm(npz_files)):
        # image_path = os.path.join(args.input_image_dir, os.path.basename(npz_file).replace("_out.npz", ""))
        image_path = os.path.join(args.input_image_dir, os.path.basename(input_npz_files[index].strip().replace(".npz", ".jpg")))

        result = decoder.npz_decode(image_path, npz_file, save_dir=args.save_dir, draw_image=args.draw_image)
