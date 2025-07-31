# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import argparse
import os

import numpy as np

parse = argparse.ArgumentParser(description="IMAGENET TOPK")
parse.add_argument("npz_txt", type=str,help = "vamp datalist")
parse.add_argument("npz_dir", type=str, help = "vamp path_output")

args = parse.parse_args()
print(args)


def decode():
    with open(args.npz_txt) as f:
        input_npz_list = [cls.strip() for cls in f.readlines()]
    output_npz_list = sorted(os.listdir(args.npz_dir))

    for file, result in zip(input_npz_list,output_npz_list):
        pred = np.load(os.path.join(args.npz_dir, result), allow_pickle=True)["output_0"].squeeze()
        print(f"{file} => {pred}")


if __name__ == "__main__":
    decode()
