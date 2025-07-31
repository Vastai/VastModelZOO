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
parse.add_argument("result_txt", type=str, help = "decode npz result")
parse.add_argument("label_txt", type=str, help="imagenet label")
args = parse.parse_args()
print(args)


def decode():
    fin = open(args.result_txt,"w")
    topk = 5
    with open(args.label_txt) as f:
        imagenet_classes = [cls.strip() for cls in f.readlines()]

    with open(args.npz_txt) as f:
        input_npz_list = [cls.strip() for cls in f.readlines()]

    output_npz_list = sorted(os.listdir(args.npz_dir))

    for file,result in zip(input_npz_list,output_npz_list):
        pred = np.load(os.path.join(args.npz_dir,result), allow_pickle=True)["output_0"]
        data_list = np.squeeze(pred)
        cls_ids = data_list.argsort()[-(topk) :][::-1]
        label_list = [imagenet_classes[i] for i in cls_ids]
        score_list = data_list[cls_ids]

        print(f"{file} => {(cls_ids, label_list, score_list)}")
        for k in range(topk):
            fin.write(
                file
                + ": "
                + "Relay top-{} id: {}, prob: {:.8f}, class name: {}".format(
                    k, cls_ids[k], score_list[k], label_list[k]
                )
                + "\n"
            )

    fin.close()


if __name__ == "__main__":
    decode()
