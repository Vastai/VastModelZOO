# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import argparse
import glob
import os
import time
import torch
import numpy as np

from model_forward import FeatureExtract


def main():
    parse = argparse.ArgumentParser(description="RUN CLS WITH VACL")
    parse.add_argument(
        "--file_path", type=str, default="./Desktop/cmcc-9/hrnet/flip/img", help="img or dir  path",
    )
    ##  ./Desktop/cmcc-9/hrnet/flip/img
    ##  ./Documents/project/det_data/coco/val2017
    parse.add_argument("--model_info", type=str, default="./configs/model_info.json", help="model info")
    parse.add_argument(
        "--vdsp_params_info", type=str, default="./configs/vdsp_hrnet.json", help="vdsp op info",
    )
    parse.add_argument("--device_id", type=int, default=0, help="device id")
    parse.add_argument("--batch", type=int,default=1,help="bacth size")
    parse.add_argument("--save_dir",default="./Desktop/cmcc-9/hrnet/flip/output")

    ####   ./Desktop/cmcc-9/hrnet/flip/output
    ####   ./Desktop/hrnet_out


    args = parse.parse_args()
    print(args)


    extractr = FeatureExtract(
        model_info=args.model_info,
        vdsp_params_info=args.vdsp_params_info,
        device_id=args.device_id,
        batch_size=args.batch,
    )

    if os.path.isfile(args.file_path):
        images =  args.file_path
    else:
        images = glob.glob(args.file_path+"/*")
    time_begin = time.time()
    
    results = extractr.extract_batch(images)
    # feat = np.reshape(result[1].astype(np.float32),result[0])
    
    # output = {}
    # output["output_0"] = feat
    # np.savez('./000000197388', **output)
    # instance_heatmaps, scores = post_process(feat)
    # print(instance_heatmaps)
    # print(scores)
    # time_end = time.time()
    # fin = open("./tmp.txt",'w')

    for (image, result) in zip(images, results):
        print(f"{image} ===> {result[0]}")
        output = {}
        feat = np.reshape(result[1].astype(np.float32),result[0])
        output["output_0"] = feat
        name = image.split('/')[-1].split('.')[0]
        np.savez(os.path.join(args.save_dir,name), **output)
    #     name = image.split('/')[-1].split('.')[0]
    #     new_name = 'output_'+str(cnt)+'_0'
    #     out = result[1].astype(np.float16)
    #     # out = np.reshape(result[1].astype(np.float16),result[0])
    #     np.save(os.path.join(target_path,new_name),out)
    #     cnt+=1
    # print(
    #     f"\n{len(images)} images in {time_end - time_begin} seconds, ({len(images) / (time_end - time_begin)} images/second)\n"
    # )
    # fin.close()
if __name__ == "__main__":
    main()