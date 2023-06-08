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


def tenor2mask(tensor_data):
    MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], 
                     [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

    if len(tensor_data.shape) < 4:
        tensor_data = tensor_data.unsqueeze(0)
    if tensor_data.shape[1] > 1:
        tensor_data = tensor_data.argmax(dim=1) 

    tensor_data = tensor_data.squeeze(1).data.cpu().numpy()
    color_maps = []
    for t in tensor_data:
        tmp_img = np.zeros(tensor_data.shape[1:] + (3,))
        # tmp_img = np.zeros(tensor_data.shape[1:])
        for idx, color in enumerate(MASK_COLORMAP):
            tmp_img[t == idx] = color
        color_maps.append(tmp_img.astype(np.uint8))
    return color_maps


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="/home/simplew/dataset/sr/CelebAMask-HQ/test_img", help="hr gt image folder")
    parse.add_argument("--gt_dir", type=str, default="/home/simplew/dataset/sr/CelebAMask-HQ/test_label", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="/home/simplew/code/vamc/vamp/0.2.0/outputs/parsenet", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[512, 512], help="vamp input shape")
    parse.add_argument("--draw_dir", type=str, default="npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()
    
    import sys
    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + '../../')
    from source_code.gpen.eval.metrics import SegMetric
    
    classes = 19
    metrics = SegMetric(n_classes=classes)
    metrics.reset()

    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".png")

            # gt image
            gt_image  = Image.open(os.path.join(args.src_dir, file_name.replace(".png", ".jpg")))
            resize_image = gt_image.resize(args.input_shape, Image.BILINEAR).convert('RGB')

            label = Image.open(os.path.join(args.gt_dir, file_name)).convert("L")
            label = label.resize(args.input_shape, Image.NEAREST)
            label = np.expand_dims(label, 0)

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            heatmap = np.load(output_npz_list[i], allow_pickle=True)["output_0"].astype(np.float32)
            vacc_pred = torch.from_numpy(heatmap)
            
            # draw
            vacc_mask = tenor2mask(vacc_pred)[0]
            cv2.imwrite(os.path.join(args.draw_dir, file_name), vacc_mask[:,:,::-1])
            
            # to eval
            pred = vacc_pred.data.max(1)[1].cpu().numpy()

            # eval metrics
            try:
                metrics.update(label, pred)
            except:
                continue

        score = metrics.get_scores()[0]
        class_iou = metrics.get_scores()[1]

        print("----------------- Total Performance --------------------")
        for k, v in score.items():
            print(k, v)

        print("----------------- Class IoU Performance ----------------")
        facial_names = ['background', 'skin', 'nose', 'eyeglass', 'left_eye', 'right_eye', 'left_brow', 'right_brow',
                        'left_ear', 'right_ear', 'mouth', 'upper_lip', 'lower_lip', 'hair', 'hat', 'earring', 'necklace',
                        'neck', 'cloth']
        for i in range(classes):
            print(facial_names[i] + "\t: {}".format(str(class_iou[i])))
        print("--------------------------------------------------------")
        ######################################################################################################


""" 
parsenet-fp16-none-3_512_512-vacc
----------------- Total Performance --------------------
Overall Acc:     0.8817983646230725
Mean Acc :       0.7166456149993303
FreqW Acc :      0.7884072970029888
Mean IoU :       0.6291738244579301
Overall F1:      0.7402903303623257
----------------- Class IoU Performance ----------------
background      : 0.8082006840259113
skin    : 0.8693285103752755
nose    : 0.8531126797915105
eyeglass        : 0.5357811304656037
left_eye        : 0.7573749495907478
right_eye       : 0.7497484082217739
left_brow       : 0.7065078237959249
right_brow      : 0.7016383069492045
left_ear        : 0.7169444765630976
right_ear       : 0.6767040865091486
mouth   : 0.7794583055934228
upper_lip       : 0.7094314522135303
lower_lip       : 0.7782397583648155
hair    : 0.799284828163522
hat     : 0.21119725802914585
earring : 0.24967173726438574
necklace        : 0.020404565399080653
neck    : 0.6081784938908326
cloth   : 0.4230952094937399
--------------------------------------------------------

parsenet-int8-kl_divergence-3_512_512-vacc
----------------- Total Performance --------------------
Overall Acc:     0.8818653041850406
Mean Acc :       0.7096703063724734
FreqW Acc :      0.7880538692172602
Mean IoU :       0.626642887578047
Overall F1:      0.7364678351530224
----------------- Class IoU Performance ----------------
background      : 0.8085589237282944
skin    : 0.8703897566482197
nose    : 0.8499362010015361
eyeglass        : 0.5350605486136207
left_eye        : 0.7565048033449521
right_eye       : 0.7541174935119045
left_brow       : 0.7020748514766875
right_brow      : 0.6999672892573232
left_ear        : 0.7156170302077313
right_ear       : 0.6743802284638648
mouth   : 0.7813428148697694
upper_lip       : 0.7120793961528652
lower_lip       : 0.7770455507579191
hair    : 0.7971429561777851
hat     : 0.1997131274245157
earring : 0.23170228521618424
necklace        : 0.0024280286376134447
neck    : 0.607606654250897
cloth   : 0.4305469242412102
--------------------------------------------------------


"""
