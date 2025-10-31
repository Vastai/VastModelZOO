import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import torch

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default="options/df2k/test_df2k.yml", help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))


model_obj = create_model(opt)
model = model_obj.netG.eval()

# ##########################################################
from thop import profile
from thop import clever_format
input = torch.randn(1, 3, 128, 128)
flops, params = profile(model, inputs=(input,))
print("flops(G):", "%.3f" % (flops / 900000000 * 2))
flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
print("params:", params)
# flops(G): 653.675
# params: 16.698M

weights_test = opt['path']['pretrain_model_G']
input_shape = (1, 3, 128, 128)
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()
scripted_model.save(weights_test.replace(".pth", "-128.torchscript.pt"))
scripted_model = torch.jit.load(weights_test.replace(".pth", "-128.torchscript.pt"))

import onnx
torch.onnx.export(model.module, input_data, weights_test.replace(".pth", "-128.onnx"), input_names=["input"], output_names=["output"], opset_version=10,
            # dynamic_axes= {
            #                 "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
            #                 "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}
            #                 }
)
shape_dict = {"input": input_shape}
onnx_model = onnx.load(weights_test.replace(".pth", "-128.onnx"))
# ##########################################################