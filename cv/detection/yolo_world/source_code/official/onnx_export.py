# # Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import warnings
import argparse
from io import BytesIO

import onnx
import torch
from mmdet.apis import init_detector
from mmengine.config import ConfigDict
from mmengine.logging import print_log
from mmengine.utils.path import mkdir_or_exist

from easydeploy.model import DeployModel, MMYOLOBackend  # noqa E402

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)
warnings.filterwarnings(action='ignore', category=torch.jit.ScriptWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=ResourceWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./code/model_check/vit_custom_models/YOLO-World/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py', help='Config file')
    # parser.add_argument('--config', default='./projects/yolo_world/YOLO-World/configs/pretrain/yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_800ft_lvis_minival.py', help='Config file')
    parser.add_argument('--checkpoint', default='./code/model_check/vit_custom_models/weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth', help='Checkpoint file')
    # parser.add_argument('--checkpoint', default='./projects/yolo_world/YOLO-World/scripts_models_export/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth', help='Checkpoint file')
    # parser.add_argument('--custom-text', default='./projects/yolo_world/YOLO-World/data/texts/obj365v1_class_texts.json', type=str, help='custom text inputs (text json) for YOLO-World.')
    parser.add_argument(
        "--export",
        type=str,
        choices=["image", "text"],
        help="choose backbone to export"
    )
    parser.add_argument('--custom-text', default='', type=str, help='custom text inputs (text json) for YOLO-World.')
    parser.add_argument('--add-padding', action="store_true", help="add an empty padding to texts.")
    parser.add_argument(
        '--model-only', action='store_true', help='Export model only', default=True)
    parser.add_argument(
        '--work-dir', default='./code/model_check/vit_custom_models/weights', help='Path to save export model')
    parser.add_argument(
        '--img-size',
        nargs='+',
        type=int,
        default=[1280, 1280],
        help='Image size of height and width')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument(
        # '--device', default='cuda:0', help='Device used for inference')
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify onnx model by onnx-sim')
    parser.add_argument(
        '--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument(
        '--backend',
        type=str,
        default='onnxruntime',
        help='Backend for export onnx')
    parser.add_argument(
        '--pre-topk',
        type=int,
        default=1000,
        help='Postprocess pre topk bboxes feed into NMS')
    parser.add_argument(
        '--keep-topk',
        type=int,
        default=100,
        help='Postprocess keep topk bboxes out of NMS')
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.65,
        help='IoU threshold for NMS')
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.25,
        help='Score threshold for NMS')
    args = parser.parse_args()
    print(args)
    args.img_size *= 2 if len(args.img_size) == 1 else 1
    return args


def build_model_from_cfg(config_path, checkpoint_path, device):
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model


def main():
    args = parse_args()
    mkdir_or_exist(args.work_dir)
    backend = MMYOLOBackend(args.backend.lower())
    if backend in (MMYOLOBackend.ONNXRUNTIME, MMYOLOBackend.OPENVINO,
                   MMYOLOBackend.TENSORRT8, MMYOLOBackend.TENSORRT7):
        if not args.model_only:
            print_log('Export ONNX with bbox decoder and NMS ...')
    else:
        args.model_only = True
        print_log(f'Can not export postprocess for {args.backend.lower()}.\n'
                  f'Set "args.model_only=True" default.')
    if args.model_only:
        postprocess_cfg = None
        output_names = None
    else:
        postprocess_cfg = ConfigDict(
            pre_top_k=args.pre_topk,
            keep_top_k=args.keep_topk,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold)
        output_names = ['num_dets', 'boxes', 'scores', 'labels']

    if len(args.custom_text) > 0:
        # with open(args.custom_text) as f:
        #     texts = json.load(f)
        texts = [args.custom_text]
    else:
        from mmdet.datasets import CocoDataset
        texts = CocoDataset.METAINFO['classes']
    if args.add_padding:
        texts = texts + [' ']
    # texts = ['dog', 'cat', 'person', 'bus', 'bycycle', 'horse']
    texts = [texts]
    baseModel = build_model_from_cfg(args.config, args.checkpoint, args.device)
    # reparameterize text into YOLO-World
    # baseModel.reparameterize(texts)
    deploy_model = DeployModel(
        baseModel=baseModel, backend=backend, postprocess_cfg=postprocess_cfg)
    deploy_model.eval()

    fake_input = torch.randn(args.batch_size, 3, *args.img_size).to(args.device)
    # dry run
    # input_ids = torch.tensor([[49406,   323, 49407], [49406,   334, 49407], [49406,   326, 49407]])
    # ['dog', ' '] tokenizer 后：
    # {'input_ids': tensor([[49406,  1929, 49407],
    #         [49406, 49407, 49407]]), 'attention_mask': tensor([[1, 1, 1],
    #         [1, 1, 0]])}

    # text = ['fire hydrant']
    # {'input_ids': tensor([[49406,  1769,  8031,   773, 49407]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]])}
    # input_ids = torch.tensor([[49406,  1929, 49407], [49406, 49407, 49407]])
    input_ids = torch.tensor([[49406, 1769, 8031, 773, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407]])

    if args.export == 'image':
        import numpy as np
        repeat_num = 1203
        input_ids = np.repeat(input_ids.numpy(), repeats=repeat_num, axis=0)
        input_ids = torch.from_numpy(input_ids)
        print(input_ids)

    # attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    # attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    if args.export == 'image':
        attention_mask = np.repeat(attention_mask.numpy(), repeats=repeat_num, axis=0)
        attention_mask = torch.from_numpy(attention_mask)

    from mmdet.structures import DetDataSample
    data_sample = DetDataSample()
    img_meta = dict(input_ids=input_ids, attention_mask=attention_mask )
    data_sample.set_metainfo(img_meta)

    deploy_model(fake_input, data_sample.input_ids, data_sample.attention_mask)
    print('deploy_model:')
    print(deploy_model)
    deploy_model.eval()

    if args.export == 'image':
        suffix_str = '_image'
    else:
        suffix_str = '_text'

    save_onnx_path = os.path.join(
        args.work_dir,
        os.path.basename(args.checkpoint).replace('.pth', suffix_str + '.onnx'))
    
    # export onnx
    torch.onnx.export(
            deploy_model,
            (fake_input, data_sample.input_ids, data_sample.attention_mask),
            save_onnx_path,
            input_names=['images', 'input_ids', 'attention_mask'],
            output_names=output_names,
            opset_version=args.opset)


if __name__ == '__main__':
    main()
