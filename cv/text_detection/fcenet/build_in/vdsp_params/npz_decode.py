# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--gt_dir", type=str, default="./dataset/ocr/icdar2015/Challenge4/ch4_test_images", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="./code/vamc/vamp/0.2.0/outputs/dbnet", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[736, 1280], help="vamp input shape, hw")
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
    from source_code.mmocr.fce_postprocess import FCEPostProcess
    from source_code.mmocr.det_metric import DetMetric
    from source_code.mmocr.label_ops import DetLabelEncode
    postprocess_op = FCEPostProcess([8, 16, 32], box_type='quad', alpha=1.2, )
    eval_class = DetMetric()
    det_lable_encode = DetLabelEncode()
    
    result_map = {}
    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".jpg")

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            cls_0 = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)
            reg_0 = np.load(npz_file, allow_pickle=True)["output_1"].astype(np.float32)
            cls_1 = np.load(npz_file, allow_pickle=True)["output_2"].astype(np.float32)
            reg_1 = np.load(npz_file, allow_pickle=True)["output_3"].astype(np.float32)
            cls_2 = np.load(npz_file, allow_pickle=True)["output_4"].astype(np.float32)
            reg_2 = np.load(npz_file, allow_pickle=True)["output_5"].astype(np.float32)
            cls = [cls_0, cls_1, cls_2]
            reg = [reg_0, reg_1, reg_2]

            # draw
            # preds = np.expand_dims(heatmap, axis=0)
            shape_list = np.array([[720, 1280, float(736) / 720, float(1280) / 1280]]) # for post processing
            post_result = postprocess_op(cls, reg, shape_list)
            result_map[os.path.basename(file_name)] = post_result[0]
    # eval
    with open(f"{args.gt_dir}/../label.txt", 'r') as f:
        for line in tqdm(f.readlines(), desc='calc metric'):
            substr = line.strip("\n").split('\t')
            file_name = substr[0].split('/')[-1]
            lable_str = substr[1]
            gt_polyons, _, ignore_tags = det_lable_encode(lable_str)
            det_polys = result_map[file_name]
            eval_class(det_polys, gt_polyons, ignore_tags)

    metric = eval_class.get_metric()
    print("metric: ", metric)
