import os
import math
import cv2
import copy
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="/home/simplew/dataset/seg/VOCdevkit/VOC2012/JPEGImages_val", help="hr gt image folder")
    parse.add_argument("--gt_dir", type=str, default="/home/simplew/dataset/seg/VOCdevkit/VOC2012/SegmentationClass", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="/home/simplew/code/vamc/vamp/0.2.0/outputs/fcn", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[512, 512], help="vamp input shape")
    parse.add_argument("--draw_dir", type=str, default="./npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=False, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()
    
    import sys
    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + '../..')
    from source_code.awesome.score import SegmentationMetric

    metric = SegmentationMetric(21)

    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".png")

            # src image
            ori_image  = Image.open(os.path.join(args.src_dir, file_name.replace(".png", ".jpg")))

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            heatmap = np.load(output_npz_list[i], allow_pickle=True)["output_0"].astype(np.float32)
            # draw
            tvm_predict = torch.tensor(heatmap)
            predict = F.softmax(tvm_predict[0].permute(1, 2, 0),dim = -1).cpu().numpy()
            predict = cv2.resize(predict, ori_image.size, interpolation=cv2.INTER_CUBIC)
            predict = predict.argmax(axis=-1)

            colors = np.loadtxt("segmentation/fcn/source_code/awesome/voc2012_colors.txt").astype('uint8')
            color = colorize(predict.astype(np.uint8), colors)
            color.save(os.path.join(args.draw_dir, file_name))

            ########################################################################################################
            # eval
            label_path = os.path.join(args.gt_dir, file_name)
            if not os.path.exists(label_path):
                continue
            gt = Image.open(label_path)
            gt = gt.resize(size=args.input_shape)
            target = np.array(gt).astype('int32')
            target[target == 255] = -1
            target = torch.from_numpy(target).long()

            metric.update(tvm_predict, target)
            pixAcc, mIoU = metric.get()
            print("{:s}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(file_name, pixAcc * 100, mIoU * 100))
            ########################################################################################################

""" 
fcn_r50_d8_20k-int8-kl_divergence-3_512_512-vacc
validation pixAcc: 92.645, mIoU: 67.449
"""
