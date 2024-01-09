import os
import cv2
import glob
import argparse
import numpy as np
import time
import torchvision
import torch
from PIL import Image
from tqdm import tqdm
from typing import Dict, Generator, Iterable, List, Union

CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                  "traffic light",
                  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
                  "frisbee",
                  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                  "surfboard",
                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                  "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear",
                  "hair drier", "toothbrush"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

class Decoder:
    def __init__(
        self,
        model_size: Union[int, list],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45
    ) -> None:
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        if isinstance(model_size,int):
            self.model_size = [model_size, model_size]
        else:
            # h,w
            assert len(model_size) == 2 , "model_size ERROR."
            self.model_size = model_size # h,w

    def rescale(self,  boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ori_shape = self.model_size
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    def xywh2xyxy(self, x):
        # Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
        
    def non_max_suppression(self,prediction,  classes=None, agnostic=False, multi_label=False, max_det=300):
        """Runs Non-Maximum Suppression (NMS) on inference results.
        This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
        Args:
            prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
            conf_thres: (float) confidence threshold.
            iou_thres: (float) iou threshold.
            classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
            agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
            multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
            max_det:(int), max number of output bboxes.

        Returns:
            list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
        """
        conf_thres = self.conf_thres
        iou_thres = self.iou_thres
        num_classes = prediction.shape[2] - 5  # number of classes
        pred_candidates = prediction[..., 4] > conf_thres  # candidates

        # Check the parameters.
        assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
        assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

        # Function settings.
        max_wh = 4096  # maximum box width and height
        max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
        time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
        multi_label &= num_classes > 1  # multiple labels per box

        tik = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for img_idx, x in enumerate(prediction):  # image index, image inference
            x = x[pred_candidates[img_idx]]  # confidence

            # If no box remains, skip the next process.
            if not x.shape[0]:
                continue

            # confidence multiply the objectness
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
            if multi_label:
                box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
            else:  # Only keep the class with highest scores.
                conf, class_idx = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class, only keep boxes whose category is in classes.
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            num_box = x.shape[0]  # number of boxes
            if not num_box:  # no boxes kept.
                continue
            elif num_box > max_nms:  # excess max boxes' number.
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
            keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if keep_box_idx.shape[0] > max_det:  # limit detections
                keep_box_idx = keep_box_idx[:max_det]

            output[img_idx] = x[keep_box_idx]
            if (time.time() - tik) > time_limit:
                print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
                break  # time limit exceeded

        return output

    def npz_decode(self, input_image_path:str, output_npz_file:str, save_dir:str, save_image=False):
        grid = [torch.zeros(1)] * 3
        stride = [8, 16, 32]
        z = []
        file_name = os.path.basename(input_image_path)
        origin_img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        
        for i in range(3):
            reg_out = torch.Tensor(np.load(output_npz_file, allow_pickle=True)["output_"+str(i*3+0)])
            obj_out = torch.Tensor(np.load(output_npz_file, allow_pickle=True)["output_"+str(i*3+1)])
            cls_out = torch.Tensor(np.load(output_npz_file, allow_pickle=True)["output_"+str(i*3+2)])

            y = torch.cat([reg_out, obj_out.sigmoid(), cls_out.sigmoid()], 1)
            bs, _, ny, nx = y.shape
            y = y.view(bs, 1, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid[i] = torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()

            xy = (y[..., 0:2] + grid[i]) * stride[i]
            wh = torch.exp(y[..., 2:4]) * stride[i]
            y = torch.cat((xy, wh, y[..., 4:]), -1)
            z.append(y.view(bs, -1, 85))
        
        z = torch.cat(z, 1)
        det = self.non_max_suppression(z, None, False, max_det=300)[0]
        
        fin = open(f"{save_dir}/{os.path.splitext(file_name)[0]}.txt", "w")
        if len(det):
            det[:, :4] = self.rescale(det[:, :4], origin_img.shape).round()

            for box in det:
                # print(box)
                label = CLASSES[int(float(box[5]))]
                score = float(box[4])
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(origin_img, p1, p2, (128, 128, 128), thickness=2, lineType=cv2.LINE_AA)
                text = f"{label}: {round(score * 100, 2)}%"
                fin.write(f"{label} {score} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n")

                ny = int(int(box[1])) - 15 if int(int(box[1])) - 15 > 15 else int(int(box[1])) + 15
                cv2.putText(
                    origin_img,
                    text,
                    (int(box[0]), ny),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLORS[int(float(box[5]))],
                    2,
                )
        if save_image:
            cv2.imwrite(f"{save_dir}/{file_name}", origin_img)
        fin.close()


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="vamp out decoding")
    parse.add_argument(
        "--input_image_dir",
        type=str,
        default="/home/rzhang/Documents/project/det_data/coco/val2017",
        help="input source image folder",
    )
    parse.add_argument(
        "--vamp_datalist_path",
        type=str,
        default="./datalist_npz.txt",
        help="vamp datalist folder",
    )
    parse.add_argument(
        "--vamp_output_dir",
        type=str,
        default="/home/rzhang/Desktop/model_latency_npz",
        help="vamp output folder",
    )
    parse.add_argument(
        "--model_size",
        nargs='+',
        type=int,
        default=[640,640],
    )
    parse.add_argument(
        "--draw_image", type=bool, default=True, help="save the draw image"
    )
    parse.add_argument("--save_dir", type=str, default="/home/rzhang/Desktop/output", help="save_dir")
    args = parse.parse_args()

    decoder = Decoder(
        model_size=args.model_size,
        conf_thres= 0.25,
        iou_thres= 0.45
    )

    with open(args.vamp_datalist_path, 'r') as f:
        input_npz_files = f.readlines()

    npz_files = glob.glob(os.path.join(args.vamp_output_dir + "/*.npz"))
    npz_files.sort()

    for index, npz_file in enumerate(tqdm(npz_files)):
        # image_path = os.path.join(args.input_image_dir, os.path.basename(npz_file).replace("_out.npz", ""))
        image_path = os.path.join(args.input_image_dir, os.path.basename(input_npz_files[index].strip().replace(".npz", ".jpg")))

        decoder.npz_decode(image_path, npz_file, save_dir=args.save_dir, save_image = args.draw_image)
