# --------------------------------------------------------
# Re-parameterizing Your Optimizers rather than Architectures (https://arxiv.org/abs/2205.15242)
# Github source: https://github.com/DingXiaoH/RepOptimizers
# Licensed under The MIT License [see LICENSE for details]
# The training script is based on the code of Swin Transformer (https://github.com/microsoft/Swin-Transformer)
# --------------------------------------------------------

import time
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from config import get_config
from data import build_loader
from lr_scheduler import build_scheduler
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, save_latest, update_model_ema_v2, unwrap_model, load_weights
import copy
from optimizer import build_optimizer, set_weight_decay
from timm.data import create_dataset, create_loader

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('RepOpt-VGG training script built on the codebase of Swin Transformer', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--arch', default="ghost-rep", type=str, help='arch name')
    parser.add_argument('--batch-size', default=1, type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', default='./dataset/cls/ILSVRC2012_img_val', type=str, help='path to dataset')
    parser.add_argument('--scales-path', default=None, type=str, help='path to the trained Hyper-Search model')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint', default="weights/RepGhostNet-0.5x-acc66.51.pth")
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],  #TODO Note: use amp if you have it
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment', default="test")
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only', default=True)
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    # dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.ARCH}")

    if 'vgg' in config.MODEL.ARCH.lower() or 'B1' in config.MODEL.ARCH or 'B2' in config.MODEL.ARCH or 'L1' in config.MODEL.ARCH or 'L2' in config.MODEL.ARCH:

        from repoptimizer.repoptvgg_model import RepOptVGG

        if 'B1' in config.MODEL.ARCH:
            num_blocks = [4, 6, 16, 1]
            width_multiplier = [2, 2, 2, 4]
        elif 'B2' in config.MODEL.ARCH:
            num_blocks = [4, 6, 16, 1]
            width_multiplier = [2.5, 2.5, 2.5, 5]
        elif 'L1' in config.MODEL.ARCH:
            num_blocks = [8, 14, 24, 1]
            width_multiplier = [2, 2, 2, 4]
        elif 'L2' in config.MODEL.ARCH:
            num_blocks = [8, 14, 24, 1]
            width_multiplier = [2.5, 2.5, 2.5, 5]
        else:
            raise ValueError('Not yet supported. You may add the architectural settings here.')

        if '-hs' in config.MODEL.ARCH:
            assert config.DATA.DATASET == 'cf100'
            model = RepOptVGG(num_blocks=num_blocks, num_classes=100, width_multiplier=width_multiplier, mode='hs')
            optimizer = build_optimizer(config, model)
        elif '-repvgg' in config.MODEL.ARCH:
            #   as baseline
            assert config.DATA.DATASET == 'imagenet'
            model = RepOptVGG(num_blocks=num_blocks, width_multiplier=width_multiplier, mode='repvgg', num_classes=1000)
            optimizer = build_optimizer(config, model)
        elif '-target' in config.MODEL.ARCH:
            assert config.DATA.DATASET == 'imagenet'
            #   build target model
            if config.EVAL_MODE or '-norepopt' in config.MODEL.ARCH:
                model = RepOptVGG(num_blocks=num_blocks, width_multiplier=width_multiplier, mode='target', num_classes=1000)
                optimizer = build_optimizer(config, model)  # just a placeholder for testing or the ablation study with regular optimizer for training
            else:
                from repoptimizer.repoptvgg_impl import build_RepOptVGG_and_SGD_optimizer_from_pth
                model, optimizer = build_RepOptVGG_and_SGD_optimizer_from_pth(num_blocks, width_multiplier, config.TRAIN.SCALES_PATH,
                                            lr=config.TRAIN.BASE_LR, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, weight_decay=config.TRAIN.WEIGHT_DECAY,
                                            num_classes=1000)
        else:
            raise ValueError('not supported')

    elif 'ghost' in config.MODEL.ARCH.lower():
        from repoptimizer.repoptghostnet_model import repoptghostnet_0_5x
        from repoptimizer.repoptghostnet_impl import build_RepOptGhostNet_0_5x_and_SGD_optimizer_from_pth

        if '-hs' in config.MODEL.ARCH:
            assert config.DATA.DATASET == 'cf100'
            model = repoptghostnet_0_5x(mode='hs', num_classes=100)
            optimizer = build_optimizer(config, model)

        elif '-rep' in config.MODEL.ARCH:
            #   as baseline
            assert config.DATA.DATASET == 'imagenet'
            model = repoptghostnet_0_5x(mode='rep')
            optimizer = build_optimizer(config, model)

        elif '-target' in config.MODEL.ARCH:
            assert config.DATA.DATASET == 'imagenet'
            #   build target model
            if config.EVAL_MODE or '-norepopt' in config.MODEL.ARCH:
                model = repoptghostnet_0_5x(mode='target', num_classes=1000)
                optimizer = build_optimizer(config, model)  # just a placeholder for testing or the ablation study with regular optimizer for training
            else:
                from repoptimizer.repoptghostnet_impl import build_RepOptGhostNet_0_5x_and_SGD_optimizer_from_pth
                model, optimizer = build_RepOptGhostNet_0_5x_and_SGD_optimizer_from_pth(config.TRAIN.SCALES_PATH,
                                            lr=config.TRAIN.BASE_LR, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, weight_decay=config.TRAIN.WEIGHT_DECAY,
                                            num_classes=1000)

        else:
            raise ValueError('not supported')

    else:
        raise ValueError('TODO: support other models except for RepOpt-VGG and RepOpt-GhostNet.')

    logger.info(str(model))
    # model.cuda()

    if torch.cuda.device_count() > 1:
        if config.AMP_OPT_LEVEL != "O0":
            model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK],
                                                          broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        if config.AMP_OPT_LEVEL != "O0":
            model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
        model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    if config.EVAL_MODE:
        load_weights(model, config.MODEL.RESUME)
        ###################################################################################################################
        model.eval()

        from thop import profile
        from thop import clever_format
        input = torch.randn(1, 3, 224, 224)
        flops, params = profile(model, inputs=(input,))
        print("flops(G):", "%.3f" % (flops / 900000000 * 2))
        flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
        print("params:", params)

        input_shape = (1, 3, 224, 224) # nchw
        shape_dict = [("input", input_shape)]
        input_data = torch.randn(input_shape)
        with torch.no_grad():
            scripted_model = torch.jit.trace(model, input_data)#.eval()
            scripted_model.save(config.MODEL.RESUME.replace(".pth", ".torchscript.pt"))
            scripted_model = torch.jit.load(config.MODEL.RESUME.replace(".pth", ".torchscript.pt"))

        # onnx==10.0.0，opset 11
        # RuntimeError: Exporting the operator hardsigmoid to ONNX opset version 11 is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub.
        with torch.no_grad():
            torch.onnx.export(model, input_data, config.MODEL.RESUME.replace(".pth", "_dynamic.onnx"), input_names=["input"], output_names=["output"], opset_version=11,
            dynamic_axes= {
                        "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
                        "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
            )
        ###################################################################################################################

import os

if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = -1
        world_size = -1
    # torch.cuda.set_device(config.LOCAL_RANK)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # torch.distributed.barrier()
    seed = config.SEED# + dist.get_rank()

    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if not config.EVAL_MODE:
        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 256.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 256.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 256.0
        # gradient accumulation also need to scale the learning rate
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()

    print('==========================================')
    print('real base lr: ', config.TRAIN.BASE_LR)
    print('==========================================')

    os.makedirs(config.OUTPUT, exist_ok=True)

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0 , name=f"{config.MODEL.ARCH}")

    # if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
    #     path = os.path.join(config.OUTPUT, "config.json")
    #     with open(path, "w") as f:
    #         f.write(config.dump())
    #     logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)


"""
python main_repopt.py --arch RepOpt-VGG-B1-target --tag test --eval --resume weights/RepOpt-VGG-B1-acc78.62.pth --data-path /path/to/imagenet --batch-size 32 --opts DATA.DATASET imagenet
python main_repopt.py --arch RepOpt-VGG-B2-target --tag test --eval --resume weights/RepOpt-VGG-B2-acc79.68.pth --data-path /path/to/imagenet --batch-size 32 --opts DATA.DATASET imagenet
python main_repopt.py --arch RepOpt-VGG-L1-target --tag test --eval --resume weights/RepOpt-VGG-L1-acc79.82.pth --data-path /path/to/imagenet --batch-size 32 --opts DATA.DATASET imagenet
python main_repopt.py --arch RepOpt-VGG-L2-target --tag test --eval --resume weights/RepOpt-VGG-L2-acc80.47.pth --data-path /path/to/imagenet --batch-size 32 --opts DATA.DATASET imagenet

python main_repopt.py --data-path /path/to/imagenet --arch ghost-rep --batch-size 128 --tag reproduce --opts TRAIN.EPOCHS 300 TRAIN.BASE_LR 0.6 TRAIN.WEIGHT_DECAY 1e-5 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 DATA.DATASET imagenet TRAIN.OPTIMIZER.NAME sgd TRAIN.WARMUP_LR 1e-4
## python main_repopt.py --data-path /path/to/imagenet --arch ghost-target --batch-size 128 --tag reproduce --scales-path weights/RepGhostNet-0.5x-acc66.51.pth --opts TRAIN.EPOCHS 300 TRAIN.BASE_LR 0.6 TRAIN.WEIGHT_DECAY 1e-5 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 DATA.DATASET imagenet TRAIN.OPTIMIZER.NAME sgd TRAIN.WARMUP_LR 1e-4


weights/RepOpt-VGG-B1-acc78.62.pth
flops(G): 26.281
params: 51.842M

fp16
[VACC]:  top1_rate: 78.002 top5_rate: 93.978
int8-percentile
[VACC]:  top1_rate: 75.956 top5_rate: 92.88

==============================================================
weights/RepOpt-VGG-B2-acc79.68.pth
flops(G): 40.866
params: 80.331M

fp16
[VACC]:  top1_rate: 79.034 top5_rate: 94.49
int8-percentile
[VACC]:  top1_rate: 76.09 top5_rate: 93.024

==============================================================
weights/RepOpt-VGG-L1-acc79.82.pth
flops(G): 46.850
params: 76.038M

fp16
[VACC]:  top1_rate: 79.266 top5_rate: 94.556
int8-percentile
[VACC]:  top1_rate: 59.652 top5_rate: 82.96
int8 kl_divergence
[VACC]:  top1_rate: 58.944 top5_rate: 82.52
==============================================================

weights/RepOpt-VGG-L2-acc80.47.pth
flops(G): 73.001
params: 118.133M

fp16
[VACC]:  top1_rate: 79.992 top5_rate: 94.918
int8-percentile
[VACC]:  top1_rate: 65.41 top5_rate: 87.02

==============================================================
weights/RepGhostNet-0.5x-acc66.51.pth
flops(G): 0.109
params: 2.314M

fp16
[VACC]:  top1_rate: 0.066 top5_rate: 0.464
int8-percentile
[VACC]:  top1_rate: 0.07 top5_rate: 0.432
"""
