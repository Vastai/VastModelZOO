# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--gt_dir", type=str, default="/path/to/dataset/ocr/icdar2015/Challenge4/ch4_test_images", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="/path/to/code/vamc/vamp/0.2.0/outputs/east", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[704, 1280], help="vamp input shape, hw")
    parse.add_argument("--draw_dir", type=str, default="./npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()
    
    import sys
    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + '../..')
    from source_code.ppocr.ct_postprocess import CTPostProcess
    from source_code.ppocr.ct_metric import CTMetric
    # from source_code.ppocr.label_ops import CTLabelEncode
    postprocess_op = CTPostProcess()
    eval_class = CTMetric()
    # ct_lable_encode = CTLabelEncode()
    
    result_map = {}
    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".jpg")
            ori_image = cv2.imread(os.path.join(args.gt_dir, file_name))

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            preds = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)

            # draw
            shape_list = [list(np.array(ori_image).shape) + [640, 960, 3]] # for post processing

            post_result = postprocess_op(preds, shape_list)
            result_map[os.path.basename(file_name)] = post_result[0]
    # eval
    with open(f"{args.gt_dir}/../test_ppocr.txt", 'r') as f:
        for line in tqdm(f.readlines(), desc='calc metric'):
            substr = line.strip("\n").split('\t')
            file_name = substr[0].split('/')[-1]

            lable_str = eval(substr[1])
            label = []
            text = []
            for lable_str_ in lable_str:
                # print(lable_str_)
                if lable_str_["transcription"] == '###':
                    continue
                text.append(lable_str_["transcription"])
                label.append(np.array(lable_str_["points"]).reshape((1, -1)))
            
            det_polys = result_map[file_name]

            eval_class(det_polys, [label, text])

    metric = eval_class.get_metric()
    print("metric: ", metric)
    ########################################################################################################

'''

# torch
[2023/08/17 02:12:19] ppocr INFO: metric eval ***************
[2023/08/17 02:12:19] ppocr INFO: total_num_gt:2215
[2023/08/17 02:12:19] ppocr INFO: total_num_det:1880
[2023/08/17 02:12:19] ppocr INFO: global_accumulative_recall:1629.1999999999962
[2023/08/17 02:12:19] ppocr INFO: hit_str_count:0
[2023/08/17 02:12:19] ppocr INFO: recall:0.7355304740406303
[2023/08/17 02:12:19] ppocr INFO: precision:0.8487234042553177
[2023/08/17 02:12:19] ppocr INFO: f_score:0.7880831935002219
[2023/08/17 02:12:19] ppocr INFO: seqerr:1.0
[2023/08/17 02:12:19] ppocr INFO: recall_e2e:0.0
[2023/08/17 02:12:19] ppocr INFO: precision_e2e:0.0
[2023/08/17 02:12:19] ppocr INFO: f_score_e2e:0
[2023/08/17 02:12:19] ppocr INFO: fps:29.680973186073665


# VACC-int8
metric:  {'total_num_gt': 2215, 'total_num_det': 2081, 'global_accumulative_recall': 1615.5999999999965, 'hit_str_count': 0, 'recall': 0.7293905191873573, 'precision': 0.7517539644401711, 'f_score': 0.7404034116661642, 'seqerr': 1.0, 'recall_e2e': 0.0, 'precision_e2e': 0.0, 'f_score_e2e': 0}
'''
