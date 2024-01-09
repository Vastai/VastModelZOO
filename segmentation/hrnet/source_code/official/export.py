import torch
import torchvision
import argparse
import _init_paths
from config import config
from config import update_config
import models
from utils.utils import create_logger, FullModel
from onnxruntime.datasets import get_example
import onnxruntime
from onnx import shape_inference
import os
from torch.nn import functional as F
import cv2
import numpy as np

def jit_export(model, pth_file):
    pretrained_dict = torch.load(pth_file, map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    
    dump_input = torch.rand(
            (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
            )
    print(model)
    print(dump_input.shape)

    traced_script_module = torch.jit.trace(model, dump_input)
    export_file = os.path.join("weights", os.path.basename(pth_file).replace(".pth",".torchscript.pt"))
    traced_script_module.save(export_file)
    
    new_model = torch.jit.load(export_file)
    dump_input = torch.rand(
            (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
            )
    out = new_model(dump_input)
    print(out)

def onnx_export(model, pth_file):
    pretrained_dict = torch.load(pth_file, map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    
    dump_input = torch.rand(
            (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
            )

    export_onnx_file = os.path.join("weights",os.path.basename(args.pth_file).replace("pth","onnx"))

    torch.onnx.export(model.cpu(), dump_input.cpu(), export_onnx_file, verbose=True, opset_version=10, input_names=["input"], output_names=["output"])
    
    dump_input = torch.rand(
            (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
            )

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    model.eval()
    x = torch.randn(1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0]).cpu()
    with torch.no_grad():
        torch_out = model(x)
    example_model = get_example(os.getcwd()+'/'+export_onnx_file)

    sess = onnxruntime.InferenceSession(example_model)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: to_numpy(x)})

    # print(torch_out.shape,torch_out[0,0,0,0:10])
    # print(onnx_out[0].shape,onnx_out[0][0,0,0,0:10])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default="/home/simplew/code/model_check/temp/HRNet-Semantic-Segmentation/experiments/lip/seg_hrnet_w48_473x473_sgd_lr7e-3_wd5e-4_bs_40_epoch150.yaml")
    parser.add_argument('--pth_file',type=str, default="/home/simplew/code/model_check/temp/HRNet-Semantic-Segmentation/weights/hrnet_w48_lip_cls20_473x473_pytorch-v11.pth")
    parser.add_argument('--image_path',type=str, default="/home/simplew/dataset/seg/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png")
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)


    args = parser.parse_args()
    update_config(config, args)
    pth_file = args.pth_file
    image_path = args.image_path

    model = eval('models.'+config.MODEL.NAME +'.get_seg_model')(config)
    model.to("cpu")
    onnx_export(model, pth_file)
    jit_export(model, pth_file)

