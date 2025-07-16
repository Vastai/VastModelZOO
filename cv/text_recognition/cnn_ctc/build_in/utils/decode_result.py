# ==============================================================================
#
# Copyright (C) 2023 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :     vastml
@Time  : 2023/04/25 17:30:50
'''
import argparse
import os

import numpy as np
import torch
from post_process import postprocess
from tqdm import tqdm

parse = argparse.ArgumentParser(description="DECODE RESULT")
parse.add_argument("result_path", type=str,help = "result data npz")
parse.add_argument("result_txt", type=str, help = "decode npz result")
parse.add_argument("--num_class", type=int, default= 37, help = "model class nums")
parse.add_argument("--batch_max", type=int, default= 25, help = "batch_max_length")

args = parse.parse_args()
print(args)

def decode():
    fin = open(args.result_txt,"w")

    for file in tqdm(os.listdir(args.result_path)):
        result = os.path.join(args.result_path,file)
        info = np.load(result)["output_0"].reshape(1, args.batch_max + 1 ,args.num_class).astype(np.float32)
        preds_str, preds_prob = postprocess(torch.from_numpy(info))
        # add prefix
        name = "test/" + file.split(".")[0] + ".png"
        fin.write(name+" "+preds_str[0]+" "+str(preds_prob[0].numpy())+ "\n")
    fin.close()


if __name__ == "__main__":
    decode()