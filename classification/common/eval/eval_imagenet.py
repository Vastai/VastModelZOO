'''
Copyright (C) 2022-2023 VASTAI Technologies Co., Ltd. All Rights Reserved.

The information contained herein is confidential property of the company.
The user, copying, transfer or disclosure of such information is prohibited
except by express written agreement with VASTAI Technologies Co., Ltd.
'''
import os
import glob
import argparse
import numpy as np

parse = argparse.ArgumentParser(description="IMAGENET TOPK")
parse.add_argument("--result_path", type=str, help="outpath of vamp")
parse.add_argument("--datalist", type=str, help="input data file list")
parse.add_argument("--label", type=str, help="label txt")
args = parse.parse_args()
print(args)

def is_match_class(topk_index, label):
    ans = []
    for i in topk_index:
        ans.append(i == label)
    return ans


def imagenet_topk(gt_label, npy_dir, input_list,  topk=5):
    """获取npy的top1&top5"""
    total_count = 0
    top1_count = 0
    top5_count = 0
    results = glob.glob(npy_dir+'/*')
    idx2label = []
    out2label = {}
    results.sort()

    # vamp输出list
    with open(input_list, 'r') as f:
        for line in f:
            idx2label.append(line.split('/')[-1].split('.')[0])
    with open(gt_label, 'r') as f:
        for line in f:
            name, lb = line.strip().split()
            out2label[name.split('.')[0]] = int(lb)

    
    for idx, result in enumerate(results):
        total_count += 1
        output = np.load(result, allow_pickle=True)["output_0"][0]
        topk_index = np.argsort(output)[::-1][:topk]

        matches = is_match_class(topk_index, out2label[idx2label[idx]])
        if matches[0]:
            top1_count += 1
            top5_count += 1
        elif True in matches:
            top5_count += 1
    top1_rate = top1_count / total_count * 100
    top5_rate = top5_count / total_count * 100
    print("[VACC]: ", "top1_rate:", top1_rate, "top5_rate:", top5_rate)
    return top1_rate, top5_rate


if __name__ == "__main__":
    imagenet_topk(args.label,args.result_path, args.datalist )

