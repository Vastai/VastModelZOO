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


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="data/ECSSD/image", help="hr gt image folder")
    parse.add_argument("--gt_dir", type=str, default="data/ECSSD/mask", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="data/ECSSD/npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="/home/simplew/code/vamc/vamp/0.2.0/outputs/f3net", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[512, 512], help="vamp input shape")
    parse.add_argument("--draw_dir", type=str, default="npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()

    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".png")

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            heatmap = np.load(output_npz_list[i], allow_pickle=True)["output_0"].astype(np.float32)
            
            # draw
            out = np.squeeze(heatmap)
            out = torch.from_numpy(out)
            pred = (torch.sigmoid(out) * 255).cpu().numpy()
            
            save_path = os.path.join(args.draw_dir, os.path.splitext(os.path.basename(file_name))[0] + ".png")
            cv2.imwrite(save_path, np.round(pred))



"""
ECSSD dataset
eval based on https://github.com/lartpang/PySODEvalToolkit

model-fp16-none-3_512_512-vacc
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.052 |         0.908 |         0.894 |         0.896 |          0.981 |          0.925 |           1 |       0.873 |   0.929 |   0.917 |   0.919 | 0.895 | 0.866 |

model-int8-kl_divergence-3_512_512-vacc
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.048 |          0.91 |           0.9 |         0.902 |          0.981 |           0.93 |           1 |       0.878 |   0.931 |   0.921 |   0.924 | 0.898 | 0.875 |
"""