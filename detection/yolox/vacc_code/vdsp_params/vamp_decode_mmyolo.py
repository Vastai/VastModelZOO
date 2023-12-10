import os
import cv2
import glob
import torch
import torchvision
import numpy as np
import argparse
from tqdm import tqdm
import shutil

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


def postprocess(prediction, num_classes, conf_thre=0.25, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    return output

def rescale(boxes, model_size, target_shape):
    '''Rescale the output to the original image shape'''
    ori_shape = model_size
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

def npz_decode(input_image_path, output_npz_file, modelsize, save_dir, save_image=False):
    filename = os.path.basename(input_image_path)
    origin_img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

    hw = [(80, 80), (40, 40), (20, 20)] 
    s = [8, 16, 32]
    grids = []
    strides = []
    outputs = []
    for i in range(3):
        cls_out = torch.Tensor(np.load(output_npz_file, allow_pickle=True)["output_"+str(i+0)])
        box_out = torch.Tensor(np.load(output_npz_file, allow_pickle=True)["output_"+str(i+3)])
        obj_out = torch.Tensor(np.load(output_npz_file, allow_pickle=True)["output_"+str(i+6)])

        output = torch.cat(
                [box_out, obj_out.sigmoid(), cls_out.sigmoid()], 1
            )

        outputs.append(output)
    for (hsize, wsize), stride in zip(hw, s):
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1)
    strides = torch.cat(strides, dim=1)
    outputs = torch.cat(
            [x.flatten(start_dim=2) for x in outputs], dim=2
        ).permute(0, 2, 1)

    outputs = torch.cat([
        (outputs[..., 0:2] + grids) * strides,
        torch.exp(outputs[..., 2:4]) * strides,
        outputs[..., 4:]
    ], dim=-1)
    pred = postprocess(outputs, 80)[0]
        
    fin = open(f"{save_dir}/{os.path.splitext(filename)[0]}.txt", "w")
    if pred is not None:
        
        pred[:, :4] = rescale(pred[:, :4], modelsize, origin_img.shape).round()

        for box in pred:
            label = CLASSES[int(float(box[6]))]
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
            cv2.imwrite(f"{save_dir}/{filename}", origin_img)
    fin.close()


if __name__=='__main__':
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

    if isinstance(args.model_size, int):
        model_size = [args.model_size, args.model_size]
    else:
        # h,w
        assert len(args.model_size) == 2 , "model_size ERROR."
        model_size = args.model_size # h,w

    with open(args.vamp_datalist_path, 'r') as f:
        input_npz_files = f.readlines()
    npz_files = glob.glob(os.path.join(args.vamp_output_dir + "/*.npz"))
    npz_files.sort()
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir,mode=0o777, exist_ok=True)
        
    
    for index, npz_file in enumerate(tqdm(npz_files)):
        image_path = os.path.join(args.input_image_dir, os.path.basename(.strip().replace(".npz", ".jpg")))
        npz_decode(image_path, npz_file, modelsize=model_size, save_dir=args.save_dir, save_image = args.draw_image)


