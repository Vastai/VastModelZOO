# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import sys
import argparse
import torch

from pathlib import Path
current_path = Path(__file__).resolve()
parent_path = current_path.parent.parent.as_posix()
sys.path.insert(0, parent_path)
import hubconf


class Wrapper(torch.nn.Module):
    """
    Wrapper class for DINOV2 model.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tensor):
        ff = self.model(tensor)
        return ff


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="DINOV2 model converter to ONNX.")
    parser.add_argument("--model_name", type=str, default="dinov2_vitl14_reg4", help="DINOV2 model name")
    parser.add_argument("--image_height", type=int, default=224, help="Input image height (multiple of patch_size)")
    parser.add_argument("--image_width", type=int, default=224, help="Input image width (multiple of patch_size)")
    parser.add_argument("--patch_size", type=int, default=16, help="DINOV2 model patch size (default is 14)")
    parser.add_argument("--simplify", action="store_true", default=True, help="Enable ONNX model simplification")
    return parser.parse_args()


def main():
    """
    Main function to convert DINOV2 model to ONNX format.
    """
    args = parse_arguments()

    assert args.image_height % args.patch_size == 0, f"Image height must be a multiple of {args.patch_size}, but got {args.image_height}"
    assert args.image_width % args.patch_size == 0, f"Image width must be a multiple of {args.patch_size}, but got {args.image_height}"


    # dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')


    model = Wrapper(getattr(hubconf, args.model_name)(for_onnx=True)).to("cpu")
    model.eval()

    dummy_input = torch.rand([1, 3, args.image_height, args.image_width]).to("cpu")
    dummy_output = model(dummy_input)

    onnx_file_path = f"{args.model_name}_new.onnx"
    torch.onnx.export(model, dummy_input, onnx_file_path, input_names=["input"], output_names=["output"])
    print(f"Model exported to: {onnx_file_path}")

    if args.simplify:
        import onnx
        try:
            from onnxsim import simplify
        except ImportError as e:
            print("onnxsim is not installed. Please install onnxsim to simplify the model.")
            sys.exit(0)

        try:
            model_simp, check = simplify(onnx_file_path)
            if check:
                onnx.save(model_simp, onnx_file_path)
                print("ONNX model is simplified.")
        except Exception as e:
            print(f"ONNX model can not be simplified due to {e}.")


if __name__ == "__main__":
    main()
