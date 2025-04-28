
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
from torch.nn import functional as F


def draw_matting(image, mask):
    """
    image (np.uint8) shape (H,W,3)
    mask  (np.float32) range from 0 to 1, shape (H,W)
    """
    mask = 255*(1.0-mask)
    mask = np.expand_dims(mask, axis=2)
    mask = np.tile(mask, (1,1,3))
    mask = mask.astype(np.uint8)
    image_alpha = cv2.add(image, mask)
    return image_alpha


def miou(logits, targets, eps=1e-6):
	"""
	logits: (torch.float32)  shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
    # using to unet, deeplabv3+
	"""
	outputs = torch.argmax(logits, dim=1, keepdim=True).type(torch.int64)
	targets = torch.unsqueeze(targets, dim=1).type(torch.int64)
	# outputs = torch.zeros_like(logits).scatter_(dim=1, index=outputs, src=torch.tensor(1.0)).type(torch.int8)
	outputs = torch.zeros_like(logits).scatter_(dim=1, index=outputs, src=torch.ones_like(logits)).type(torch.int8)

	# targets = torch.zeros_like(logits).scatter_(dim=1, index=targets, src=torch.tensor(1.0)).type(torch.int8)
	targets = torch.zeros_like(logits).scatter_(dim=1, index=targets, src=torch.ones_like(logits)).type(torch.int8)

	inter = (outputs & targets).type(torch.float32).sum(dim=(2,3))
	union = (outputs | targets).type(torch.float32).sum(dim=(2,3))
	iou = inter / (union + eps)
	return iou.mean()


def custom_bisenet_miou(logits, targets):
	"""
	logits: (torch.float32) (main_out, feat_os16_sup, feat_os32_sup) of shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		return miou(logits[0], targets)
	else:
		return miou(logits, targets)


def custom_pspnet_miou(logits, targets):
	"""
	logits: (torch.float32) (main_out, aux_out) of shape (N, C, H, W), (N, C, H/8, W/8)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		return miou(logits[0], targets)
	else:
		return miou(logits, targets)


def custom_icnet_miou(logits, targets):
	"""
	logits: (torch.float32)
		[train_mode] (x_124_cls, x_12_cls, x_24_cls) of shape
						(N, C, H/4, W/4), (N, C, H/8, W/8), (N, C, H/16, W/16)

		[valid_mode] x_124_cls of shape (N, C, H, W)

	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		targets = torch.unsqueeze(targets, dim=1)
		targets = F.interpolate(targets, size=logits[0].shape[-2:], mode='bilinear', align_corners=True)[:,0,...]
		return miou(logits[0], targets)
	else:
		return miou(logits, targets)



if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="/path/to/Human-Segmentation-PyTorch/dataset/Supervisely_Person_Dataset/src", help="hr gt image folder")
    parse.add_argument("--gt_dir", type=str, default="/path/to/Human-Segmentation-PyTorch/dataset/Supervisely_Person_Dataset/mask", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="outputs/deeplabv3plus_resnet18", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[320, 320], help="vamp input shape, w,h")
    parse.add_argument("--draw_dir", type=str, default="./npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()
    ious = []

    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            line = line.strip()
            npz_sub_name = line.split("/")[-2] + "/" +  line.split("/")[-1]
            image_sub_name = line.split("/")[-2] + "/img/" +  line.split("/")[-1]
            image_sub_name = image_sub_name.replace(".npz", ".png")

            # src image
            ori_image  = Image.open(os.path.join(args.src_dir, image_sub_name))

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, os.path.basename(npz_sub_name))

            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)
            tvm_output = torch.from_numpy(heatmap)

            # # draw matting
            vacc_preds = F.interpolate(tvm_output, size=ori_image.size[::-1], mode='bilinear', align_corners=True)
            vacc_preds = F.softmax(vacc_preds, dim=1)
            vacc_preds = vacc_preds[0,1,...].numpy()
            try:
                image_alpha = draw_matting(np.array(ori_image), vacc_preds)
            except:
                 continue
            cv2.imwrite(os.path.join(args.draw_dir, os.path.basename(line)+ ".matting.png"), image_alpha[..., ::-1])
            
            # # draw mask
            vacc_preds = tvm_output[0].cpu().numpy()
            vacc_preds = np.asarray(vacc_preds, dtype="float32").transpose(1, 2, 0)
            mask = cv2.resize(vacc_preds, ori_image.size, interpolation=cv2.INTER_CUBIC)
            color = mask.argmax(axis=2).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(args.draw_dir, os.path.basename(line) + ".mask.png"), color)
        
            ########################################################################################################
            # eval
            # gt
            mask_path = os.path.join(args.gt_dir, image_sub_name.replace("/img", ""))
            label = cv2.imread(mask_path, 0)
            label = cv2.resize(label, args.input_shape, interpolation=cv2.INTER_LINEAR)
            label[label>0] = 1
            targets = np.expand_dims(label, axis=0)

            targets = torch.tensor(targets.copy(), dtype=torch.float32)
            iou = miou(tvm_output, targets).numpy()
            ious.append(iou)
            print('{}, --> miou: {}'.format(image_sub_name, str(iou*100)))

        mean_iou = np.mean(ious)
        print("mean iou: {}".format(mean_iou*100))
	

""" 
deeplabv3plus_resnet18-int8-kl_divergence-3_320_320-vacc
mean iou: 73.94943237304688

"""
