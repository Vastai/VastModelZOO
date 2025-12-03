# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="asdfadffuck",
        help="output dir",
    )
    args = parser.parse_args()

    output_dir=os.path.join(os.getcwd(), args.out_dir)+"/"

   
    os.path.dirname(os.path.abspath(__file__))
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_file_path)


    cmd = f"python3 int8_pointpillar_with_detection_eval.py --output_path {output_dir}"

    os.system(
        "cd kitti_eval_system/kitti_eval_src && mkdir -p build && cd build && cmake .. && make && mv evaluate_object ../.."
    )

    os.system("cd kitti_eval_system && mkdir -p results/data")

    os.system(cmd)

    os.system("cd kitti_eval_system && ./evaluate_object  ./label ./results/")

