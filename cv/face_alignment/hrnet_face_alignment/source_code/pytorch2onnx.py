# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import pprint
import argparse

import torch
import torch.backends.cudnn as cudnn
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    # load model
    # state_dict = torch.load(args.model_file)
    state_dict = torch.load(args.model_file, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    
    ## export onnx
    img = torch.randn(1, 3, 256, 256)
    input_names = ["input"]
    torch.onnx.export(model,
                img,
                "./hrnetv2.onnx",
                verbose=True,
                opset_version = 11,
                input_names=input_names,
                dynamic_axes={args.input: {0: 'batch'},
                    args.output: {0: 'batch'}},
    )


if __name__ == '__main__':
    main()

