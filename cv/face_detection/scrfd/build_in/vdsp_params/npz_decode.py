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

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.
    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.
    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def nms(dets):
    thresh = 0.45
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def post_process(net_outs, det_scale, img, thresh=0.5, use_kps=True, input_height=640, input_width=640):
    scores_list = []
    bboxes_list = []
    kpss_list = []
    fmc = 3
    feat_stride_fpn = [8, 16, 32]
    num_anchors =2
    max_num =0
    metric='default'
    for idx, stride in enumerate(feat_stride_fpn):
        # If model support batch dim, take first output

        scores = net_outs[idx]
        bbox_preds = net_outs[idx + fmc]
        bbox_preds = bbox_preds * stride
        if use_kps:
            kps_preds = net_outs[idx + fmc * 2] * stride

        height = input_height // stride
        width = input_width // stride
        K = height * width
        key = (height, width, stride)

        #solution-3:
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        #print(anchor_centers.shape)

        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if num_anchors>1:
            anchor_centers = np.stack([anchor_centers]*num_anchors, axis=1).reshape((-1,2))

        pos_inds = np.where(scores>=thresh)[0]
        #bbox_preds = bbox_preds.reshape((-1,4))
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)
        if use_kps:
            kpss = distance2kps(anchor_centers, kps_preds)
            #kpss = kps_preds
            kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

    #return scores_list, bboxes_list, kpss_list
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    #bboxes = bboxes_list
    # bboxes = np.vstack(bboxes_list) / det_scale
    bboxes_list = np.vstack(bboxes_list)
    bboxes_list[:, ::2] = bboxes_list[:, ::2] / det_scale[0]
    bboxes_list[:, 1::2] = bboxes_list[:, 1::2] / det_scale[1]
    bboxes = bboxes_list
    if use_kps:
        # kpss = np.vstack(kpss_list) / det_scale
        kpss_list = np.vstack(kpss_list)
        kpss_list[:, ::2] = kpss_list[:, ::2] / det_scale[0]
        kpss_list[:, 1::2] = kpss_list[:, 1::2] / det_scale[1]
        kpss = kpss_list
    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = nms(pre_det)
    det = pre_det[keep, :]
    if use_kps:
        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]
    else:
        kpss = None

    if max_num > 0 and det.shape[0] > max_num:
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
        img_center = img.shape[0] // 2, img.shape[1] // 2
        offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        if metric=='max':
            values = area
        else:
            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
        bindex = np.argsort(
            values)[::-1]  # some extra weight on the centering
        bindex = bindex[0:max_num]
        det = det[bindex, :]
        if kpss is not None:
            kpss = kpss[bindex, :]
    return det, kpss

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

    def postprocess(self, out0, out1, out2, out3, out4, out5, classes_list, image_file, save_dir, save_img=False, **kwargs):
        origin_img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        h, w, c = origin_img.shape

        file_name = os.path.basename(image_file)
        os.makedirs(save_dir, exist_ok=True)

        net_outs = [out0, out1, out2, out3, out4, out5]
        for i in range(3):
            net_outs[i] = torch.from_numpy(net_outs[i]).float()
            net_outs[i] = net_outs[i].permute(0, 2, 3, 1).reshape(-1, 1).sigmoid()
        for i in range(3, 6):
            net_outs[i] = torch.from_numpy(net_outs[i])
            net_outs[i] = net_outs[i].permute(0, 2, 3, 1).reshape(-1, 4)

        for i in range(6):
            net_outs[i] = net_outs[i].numpy()

        bboxes, kpss = post_process(net_outs, [640/w, 640/h], origin_img, use_kps=len(stream_ouput_data)==9)

        COLORS = np.random.uniform(0, 255, size=(len(classes_list), 3))
        if not os.path.isdir(os.path.join(save_dir, image_file.split('/')[-2])):
            os.makedirs(os.path.join(save_dir, image_file.split('/')[-2]))
        fin = open(os.path.join(save_dir, image_file.split('/')[-2], image_file.split('/')[-1].replace('jpg', 'txt')), 'w')
        file_name = os.path.basename(image_file)[:-4]
        fin.writelines(file_name + '\n')
        fin.write(str(bboxes.shape[0]) + '\n')
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1,y1,x2,y2,score = bbox
            box = [x1, y1, x2-x1, y2-y1, score]
            save_txt = ' '.join([str(bb) for bb in box])
            fin.writelines(save_txt + '\n')
        fin.close()

    def npz_decode(self, input_image_path: str, output_npz_file: str, txt_save_dir):
        # print(output_npz_file)
        #print(np.load(output_npz_file, allow_pickle=True).files)
        out0 = np.load(output_npz_file, allow_pickle=True)["output_0"]
        out1 = np.load(output_npz_file, allow_pickle=True)["output_1"]
        out2 = np.load(output_npz_file, allow_pickle=True)["output_2"]
        out3 = np.load(output_npz_file, allow_pickle=True)["output_3"]
        out4 = np.load(output_npz_file, allow_pickle=True)["output_4"]
        out5 = np.load(output_npz_file, allow_pickle=True)["output_5"]
        stream_ouput = (out0, out1, out2, out3, out4, out5)

        # post proecess
        self.postprocess(out0, out1, out2, out3, out4, out5,
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
