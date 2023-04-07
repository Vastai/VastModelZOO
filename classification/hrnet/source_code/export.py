'''
Copyright (C) 2022-2023 VASTAI Technologies Co., Ltd. All Rights Reserved.

The information contained herein is confidential property of the company.
The user, copying, transfer or disclosure of such information is prohibited
except by express written agreement with VASTAI Technologies Co., Ltd.
'''
import argparse
import torch
import onnx
import _init_paths
import models
from lib.config import config
from lib.models.cls_hrnet import get_cls_net

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, help='hrnet config file')
    parser.add_argument('--weight_path', type=str, help='weights path')
    parser.add_argument('--save_name', type=str, help='save weights name')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    config.defrost()

    config.merge_from_file(opt.cfg_file)
    config.TEST.MODEL_FILE = opt.weight_path

    config.freeze()
    model = get_cls_net(config)
    model.load_state_dict(torch.load(config.TEST.MODEL_FILE, map_location=torch.device('cpu')))
    model.eval()

    model_save = 'export_model/' + opt.save_name + '_' + str(224)

    input_shape = (1, 3, 224, 224)
    input_data = torch.randn(input_shape)
    model(input_data)
    scripted_model = torch.jit.trace(model, input_data)

    # scripted_model = torch.jit.script(net)
    torch.jit.save(scripted_model, model_save + '.torchscript.pt')


    # model.eval()
    input_names = ["input"]
    output_names = ["output"]
    inputs = torch.randn(1, 3, 224, 224)

    torch_out = torch.onnx._export(model, inputs, model_save + '.onnx', export_params=True, verbose=False, opset_version=10,
                                input_names=input_names, output_names=output_names)




