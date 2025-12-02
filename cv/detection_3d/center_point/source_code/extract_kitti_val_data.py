# 把val.txt的数据从train数据集中提取到单独的文件夹中

import shutil
import os
import argparse

parser = argparse.ArgumentParser(description="RUN EXTRACT")
parser.add_argument("--kitti_data", type = str, default = "kitti", help = "kitti data")
parser.add_argument("--output_dir", type = str, default = "./kitti_val_points", help = "output dir")
args = parser.parse_args()

val_txt = f"{args.kitti_data}/ImageSets/val.txt"
kitti_data = f"{args.kitti_data}/training/velodyne"
kitti_val_points = args.output_dir
if not os.path.exists(kitti_val_points):
    os.makedirs(kitti_val_points)

with open(val_txt, 'r') as file:
    val_lines = file.readlines()  # 按行读取内容
    val_lines = [line.strip() for line in val_lines]  # 去除每行的换行符
    for i in range(len(val_lines)):
        bin_path = os.path.join(kitti_data, val_lines[i] + ".bin")
        command = f"cp {bin_path} {kitti_val_points}"
        print(command)
        os.system(command)