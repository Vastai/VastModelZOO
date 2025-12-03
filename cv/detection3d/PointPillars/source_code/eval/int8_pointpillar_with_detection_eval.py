# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import numpy as np
import glob
import evals.decoder_3d as decoders
import argparse

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        default="",
        help="inference output path",
    )
    args = parser.parse_args()

    file_index = 0
    all_files = [f for f in sorted(glob.glob(args.output_path+"/*.bin.box", recursive=True))]
    for file in all_files: 
        print("begin..",file)
        sample_file = file
        sample_file = sample_file.split('/')[-1].split('.')[0]
        sample_file_score = args.output_path+"./{0}.bin.score".format(sample_file)
        sample_file_label = args.output_path+"/{0}.bin.label".format(sample_file)
        sample_file_boxes = args.output_path+"/{0}.bin.box".format(sample_file)
        
        out_class = np.fromfile(sample_file_label, np.float16).astype("int32")
        out_confidence = np.fromfile(sample_file_score, np.float16).astype("float32")
        out_boxes = np.fromfile(sample_file_boxes, np.float16).reshape(500,7).astype("float32")
 
        masks = out_class > -1
        out_confidence = out_confidence[masks]
        out_class = out_class[masks]
        out_boxes = out_boxes[masks]

        decoders.generater_results_by_output_with_detection(file_index, out_confidence, out_class, out_boxes)
        
        print("dealing...:", file_index, "--", sample_file)
        file_index = file_index + 1
