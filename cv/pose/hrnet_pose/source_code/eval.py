# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import torch
import glob
import cv2
import numpy as np
import argparse
from utils.utils import merge_configs
from data_process import data_process
from hrnet_postprocess import forward_test
from mmpose.datasets.datasets.bottom_up import BottomUpCocoDataset
from utils.coco import dataset_info as coco_datainfo
from data_process import data_cfg, test_pipeline
from tqdm import trange



def post_process(img, output,outflip ,flip_test = True):
        
    img_name = os.path.basename(img).split('.')[0]
    imgdata= cv2.imread(img)
    h, w, _ = imgdata.shape
    pred_data = []
    output_flip = []

    toutput = np.load(os.path.join(output,img_name+'.npz'),allow_pickle=True)
    pred_data.append(torch.Tensor(toutput["output_0"]))

    toutput_flip = np.load(os.path.join(outflip,img_name+'.npz'),allow_pickle=True)
    output_flip.append(torch.Tensor(toutput_flip["output_0"]))

    # _ , img_metas , sigmas= data_process(img)
    img_metas= data_process(img)
    # print(img_metas)
    # exit()
    img_metas['base_size'] = (w, h)
    ncl = max(w,h)/200
    img_metas['scale'] = np.array([ncl, ncl])
    img_metas['center'] = (int(w/2),int(h/2))

    
    result = forward_test(outputs=pred_data,
        # img=img,
        outputs_flipped = output_flip,
        img_metas=img_metas,
        return_heatmap=False,
        flip_test=flip_test
    )
    # yield result  ###
    return result


def main(arg):
    img_dir = arg.data_root+"/val2017"
    imglist = glob.glob(img_dir+"/*")

    print("Start feature postprocessing...")
    results = []

    for img, i in zip(imglist, trange(len(imglist))):
        results.append(post_process(img,args.output_data,args.output_flip))

    eval_config = dict(interval=50, metric='mAP', save_best='AP')
    eval_config = merge_configs(eval_config, dict(metric='mAP'))
    
    dataset = BottomUpCocoDataset(
        ann_file=f'{args.data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{args.data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info=coco_datainfo,
        test_mode=True)

    print("Accuracy calculation in progress")
    os.makedirs('./work_dir',mode=0o777, exist_ok=True)

    out_result = dataset.evaluate(results, './work_dir', **eval_config)
    
    for k, v in sorted(out_result.items()):
        print(f'{k}: {v}')

def parse_args():
    parser = argparse.ArgumentParser(description="Convert front model to vacc.")
    parser.add_argument('--flip_test',action="store_true" )
    parser.add_argument('--data_root',default= "./Documents/project/det_data/coco")
    parser.add_argument("--output_data",default= "./Desktop/hrnet_out")
    parser.add_argument("--output_flip",default= "./Desktop/cmcc-9/hrnet/flip/output")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)