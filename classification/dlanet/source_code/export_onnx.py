'''
Copyright (C) 2022-2023 VASTAI Technologies Co., Ltd. All Rights Reserved.

The information contained herein is confidential property of the company.
The user, copying, transfer or disclosure of such information is prohibited
except by express written agreement with VASTAI Technologies Co., Ltd.
'''
import torch
import thop
import math
import argparse
from thop import clever_format
from torch import nn
from dla import BasicBlock,Tree,Bottleneck,BottleneckX

BatchNorm = nn.BatchNorm2d

class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, return_levels=False,
                 pool_size=7, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                            stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x

    def load_pretrained_model(self, name, local_dir):
        self.load_state_dict(torch.load(local_dir))

class select_fun():
    def dla34(root,**kwargs):  # DLA-34
        model = DLA([1, 1, 1, 2, 2, 1],
                    [16, 32, 64, 128, 256, 512],
                    block=BasicBlock)
        local_dir = root+"dla34-ba72cf86.pth"
        model.load_pretrained_model('dla34',local_dir)
        return model


    def dla46_c(root,**kwargs):  # DLA-46-C
        Bottleneck.expansion = 2
        model = DLA([1, 1, 1, 2, 2, 1],
                    [16, 32, 64, 64, 128, 256],
                    block=Bottleneck)
        local_dir = root + "dla46_c-2bfd52c3.pth"
        model.load_pretrained_model('dla46_c',local_dir)
        return model


    def dla46x_c(root,**kwargs):  # DLA-X-46-C
        BottleneckX.expansion = 2
        model = DLA([1, 1, 1, 2, 2, 1],
                    [16, 32, 64, 64, 128, 256],
                    block=BottleneckX)
        local_dir  = root + "dla46x_c-d761bae7.pth"
        model.load_pretrained_model('dla46x_c',local_dir)
        return model


    def dla60x_c(root,**kwargs):  # DLA-X-60-C
        BottleneckX.expansion = 2
        model = DLA([1, 1, 1, 2, 3, 1],
                    [16, 32, 64, 64, 128, 256],
                    block=BottleneckX)
        local_dir  = root + "dla60x_c-b870c45c.pth"
        model.load_pretrained_model('dla60x_c',local_dir)
        return model


    def dla60(root,**kwargs):  # DLA-60
        Bottleneck.expansion = 2
        model = DLA([1, 1, 1, 2, 3, 1],
                    [16, 32, 128, 256, 512, 1024],
                    block=Bottleneck)
        local_dir = root + "dla60-24839fc4.pth"
        model.load_pretrained_model('dla60', local_dir)
        return model


    def dla60x(root,**kwargs):  # DLA-X-60
        BottleneckX.expansion = 2
        model = DLA([1, 1, 1, 2, 3, 1],
                    [16, 32, 128, 256, 512, 1024],
                    block=BottleneckX)
        local_dir = root + "dla60x-d15cacda.pth"
        model.load_pretrained_model('dla60x',local_dir)
        return model


    def dla102(root,**kwargs):  # DLA-102
        Bottleneck.expansion = 2
        model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                    block=Bottleneck, residual_root=True)
        local_dir = root + "dla102-d94d9790.pth"
        model.load_pretrained_model('dla102',local_dir)
        return model


    def dla102x(root,**kwargs):  # DLA-X-102
        BottleneckX.expansion = 2
        model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                    block=BottleneckX, residual_root=True)
        local_dir = root + "dla102x-ad62be81.pth"
        model.load_pretrained_model('dla102x', local_dir)
        return model


    def dla102x2(root,**kwargs):  # DLA-X-102 64
        BottleneckX.cardinality = 64
        model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                    block=BottleneckX, residual_root=True)
        local_dir = root + "dla102x2-262837b6.pth"
        model.load_pretrained_model('dla102x2',local_dir)
        return model


    def dla169(root,**kwargs):  # DLA-169
        Bottleneck.expansion = 2
        model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                    block=Bottleneck, residual_root=True)
        local_dir =root + "dla169-0914e092.pth"
        
        model.load_pretrained_model('dla169', local_dir)
        return model

def count_op(model, input):
    flops, params = thop.profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert pretrained model to onnx file.')
    parser.add_argument('--model_name',default="dla34",type=str)
    parser.add_argument('--infile',default='/home/rzhang/workspace/dla/pretrained_weight/',help="folder of pre-trained model")
    parser.add_argument('--output',default='/home/rzhang/workspace/',help="path to save onnx file")
    args = parser.parse_args()
    return args

def main(args):
    model_fun = getattr(select_fun,args.model_name)
    model = model_fun(args.infile)
    img = torch.randn(1,3,224,224)
    model.eval()
    count_op(model, img)
    torch.onnx.export(model,img,args.output+args.model_name+".onnx", input_names=["input"], opset_version=10)
    print(f"{args.model_name} export done")

if __name__  == "__main__":
    
    """
    dla34,  dla46_c, dla46x_c, dla60x_c,  dla60, 
    dla60x, dla102,  dla102x,  dla102x2,  dla169
    """
    args =parse_args()
    main(args)
   