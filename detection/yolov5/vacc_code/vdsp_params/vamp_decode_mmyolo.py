import argparse
import glob
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
from source_code.util import non_max_suppression, scale_coords, letterbox, plot_one_box, names, colors

def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def yolo_data_process_cv(img, img_shape=[640, 640]):
    img, _, _ = letterbox(img, img_shape)

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = img.astype(np.float32)
    img /= 255.0
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)
    return img


def postprocess(anchor, stride, result):
    z = []
    anchor = torch.tensor(anchor).view(3, 1, 3, 1, 1, 2)  # .cuda()

    for i in range(3):
        out = result[i]
        bs, _, ny, nx = out.shape
        grid = _make_grid(nx, ny)     # .cuda()
        out = out.view(1, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        y = out.sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor[i]  # wh
        z.append(y.view(1, -1, 85))

    o = torch.cat(z, 1)
    pred = non_max_suppression(o, 0.001, 0.65, None, False, max_det=300)
    return pred


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

        if isinstance(model_size, int):
            self.model_size = [model_size, model_size]
        else:
            # h,w
            assert len(model_size) == 2, "model_size ERROR."
            self.model_size = model_size  # h,w
        self.classes = classes
        self.threashold = threashold

    def postprocess(self, stream_ouput, classes_list, image_file, save_dir, save_img=False, **kwargs):
        origin_img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        height, width, _ = origin_img.shape
        file_name = os.path.basename(image_file)
        os.makedirs(save_dir, exist_ok=True)

        stride = [8, 16, 32]
        anchor = [10, 13, 16, 30, 33, 23, 30, 61, 62,
                  45, 59, 119, 116, 90, 156, 198, 373, 326]

        z = []
        for i in range(3):
            yolo_layer = torch.Tensor(stream_ouput[i])
            z.append(yolo_layer)

        pred = postprocess(anchor, stride, z)

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    [640, 640], det[:, :4], (height, width, 3))  # .round()

        det = det.numpy()
        res_length = len(det)
        COLORS = np.random.uniform(0, 255, size=(len(classes_list), 3))
        fin = open(f"{save_dir}/{os.path.splitext(file_name)[0]}.txt", "w")
        # 画框
        if res_length:
            for index in range(res_length):
                label = classes_list[det[index][5].astype(np.int8)]
                score = det[index][4]
                bbox = det[index][:4].tolist()
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
                    COLORS[det[index][5].astype(np.int8)],
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
        out0 = np.load(output_npz_file, allow_pickle=True)["output_0"]
        out1 = np.load(output_npz_file, allow_pickle=True)["output_1"]
        out2 = np.load(output_npz_file, allow_pickle=True)["output_2"]
        stream_ouput = [out0, out1, out2]

        # post proecess
        self.postprocess(stream_ouput,
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
    os.makedirs(txt_save_dir, mode=0o777, exist_ok=True)

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
    parse.add_argument("--format", type=str, default="bbox",
                       help="'segm', 'bbox', 'keypoints'")
    parse.add_argument("--txt", type=str,
                       default="./TEMP_TXT", help="txt files")
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
        default=[640, 640],
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
