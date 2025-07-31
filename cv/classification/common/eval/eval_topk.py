# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import argparse
import numpy as np
from imagenet_info import imagenet_info


def is_match_class(line, label_info):
    # get gt & dt
    line = line.rstrip("\n")
    if len(line) == 0:
        return False
    line_info_path = line.split("/")
    line_info_relay = line.split(":")
    gt_label_dir = line_info_path[-2]
    dt_label_name = line_info_relay[4].strip()
    gt_label_name = label_info[gt_label_dir]
    return gt_label_name == dt_label_name

def imagenet_topk(txt_path, topk=5, label_info=imagenet_info):
    """获取单个模型txt文件的top1&top5"""
    total_count = 0
    top1_count = 0
    top5_count = 0
    with open(txt_path, "r") as fout:
        lines = fout.readlines()
        for i in range(0, len(lines), topk):
            total_count += 1
            five_lines = lines[i : i + topk]
            matches = [is_match_class(line, label_info) for line in five_lines]
            if matches[0]:
                top1_count += 1
                top5_count += 1
            elif True in matches:
                top5_count += 1
    top1_rate = top1_count / total_count * 100
    top5_rate = top5_count / total_count * 100
    print("[VACC]: ", "top1_rate:", np.around(top1_rate, 3), "top5_rate:", np.around(top5_rate, 3))
    return top1_rate, top5_rate


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="IMAGENET TOPK")
    parse.add_argument("result_txt", type=str)
    args = parse.parse_args()
    print(args)
    
    imagenet_topk(txt_path = args.result_txt)

