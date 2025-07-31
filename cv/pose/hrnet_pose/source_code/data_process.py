# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import numpy as np
from utils.dataset_info import DatasetInfo
from utils.coco import dataset_info
from mmpose.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

dataset_info_new = DatasetInfo(dataset_info)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index'
        ]),
]
channel_cfg = dict(
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])
data_cfg = dict(
    image_size=512,
    base_size=256,
    base_sigma=2,
    heatmap_size=[128],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=1,
    scale_aware_sigma=False,
)

def data_process_old(img_or_path,):
    # build the data pipeline
    test_pipeline_new = Compose(test_pipeline)
    # test_pipeline_new = None
    if dataset_info is not None:
        dataset_name = dataset_info_new.dataset_name
        flip_index = dataset_info_new.flip_index
        sigmas = getattr(dataset_info_new, 'sigmas', None)
        skeleton = getattr(dataset_info_new, 'skeleton', None)
    # prepare data
    data = {
        'dataset': dataset_name,
        'ann_info': {
            'image_size': np.array(data_cfg['image_size']),
            'heatmap_size': data_cfg.get('heatmap_size', None),
            'num_joints': data_cfg['num_joints'],
            'flip_index': flip_index,
            'skeleton': skeleton,
        }
    }

    data['image_file'] = img_or_path
    data = test_pipeline_new(data)
    
    ### data transform
    data = collate([data], samples_per_gpu=1)

    data = scatter(data, [-1])[0]
    return data['img'], data['img_metas'], sigmas

def data_process(img_or_path,):
    # img_metas = []
    data = {
        'image_file': img_or_path,
        'test_scale_factor': [1],
        'flip_index': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    }
    # img_metas.append(data)
    return data


if __name__ == "__main__":
    img_or_path = "configs/000000040083.jpg"
    data_process(img_or_path)