# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from model.retinanet_forward import retinanet_from_backbone
import argparse
import numpy as np
import torch
import torchvision


def get_args():
    """
    Args used for converting PyTorch/TorchScript to ONNX model
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--weights",
        default="./code/algorithm_modelzoo/0118/algorithm_modelzoo/retinanet_model_10.pth",
        help="Path to the PyTorch model weights",
    )
    parser.add_argument(
        "--output",
        default="resnext50_32x4d_fpn_forward.onnx",
        help="Output file of the model",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Declare general model params
    args = get_args()
    backbone = "resnext50_32x4d"
    num_classes = 264
    image_size = [800, 800]

    model = retinanet_from_backbone(backbone, num_classes, image_size=image_size)
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    sample_input = torch.randn(1, 3, 800, 800)
    # import cv2
    # origin_image = cv2.imread('./code/algorithm_modelzoo/0118/algorithm_modelzoo/inference/vision/classification_and_detection/tools/inference/vision/classification_and_detection/tools/datasets/openimages/validation/data/0009bad4d8539bb4.jpg')
    # original_image_sizes = [origin_image[0], origin_image[1]] # hw
    # image = origin_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    # image = image / 255.0
    # image = np.ascontiguousarray(image).astype(np.float32)
    # image = np.expand_dims(image, axis=0)

    # sample_input = torch.from_numpy(image)

    torch.onnx.export(
        model,
        sample_input,
        args.output,
        export_params=True,
        opset_version=11,
        # output_names=["boxes", "scores", "labels"],
    )
