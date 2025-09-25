import argparse
import torch
import onnx
import _init_paths
import models
from lib.config import config
from lib.models.cls_hrnet import get_cls_net
from thop import profile
from thop import clever_format

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, help='hrnet config file')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    config.defrost()

    config.merge_from_file(opt.cfg_file)

    config.freeze()
    model = get_cls_net(config)
    # model.load_state_dict(torch.load(config.TEST.MODEL_FILE, map_location=torch.device('cpu')))
    model.eval()

    input_shape = (1, 3, 224, 224)
    input_data = torch.randn(input_shape)
    model(input_data)

    flops, params = profile(model, inputs=(input_data, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)



