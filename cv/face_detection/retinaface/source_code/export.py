import argparse
import torch
import onnx
from models.retinaface import RetinaFace

net_config = {
    "mobilenet0.25": {
        'name': 'mobilenet0.25',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 32,
        'ngpu': 1,
        'epoch': 250,
        'decay1': 190,
        'decay2': 220,
        'image_size': 640,
        'pretrain': True,
        'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
        'in_channel': 32,
        'out_channel': 64
    },
    "Resnet50": {
        'name': 'Resnet50',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 24,
        'ngpu': 4,
        'epoch': 100,
        'decay1': 70,
        'decay2': 90,
        'image_size': 840,
        'pretrain': True,
        'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
        'in_channel': 256,
        'out_channel': 256
    }
}

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=['mobilenet0.25', 'Resnet50'], help='retinaface model type')
    parser.add_argument('--weight_path', type=str, help='weights path')
    parser.add_argument('--size', type=int, nargs="+", default=[640], help='weights path')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    net = RetinaFace(cfg=net_config[opt.model_name], phase='test')
    net = load_model(net, opt.weight_path, True)
    net.eval()

    for size in opt.size:

        model_save = 'export_model/' + opt.model_name + '_' + str(size)

        input_shape = (1, 3, size, size)
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(net, input_data).eval()

        # scripted_model = torch.jit.script(net)
        torch.jit.save(scripted_model, model_save + '.torchscript.pt')

        input_names = ["input"]
        output_names = ["output"]
        inputs = torch.randn(1, 3, size, size)

        torch_out = torch.onnx._export(net, inputs, model_save + '.onnx', export_params=True, verbose=False,
                                    input_names=input_names, output_names=output_names, opset_version=10)

        '''torch.onnx.export(net, input_data, model_save + '.onnx', verbose=False, opset_version=10,
                            training=torch.onnx.TrainingMode.EVAL,
                            do_constant_folding=True,
                            input_names=['input'],
                            output_names=['output'])'''

