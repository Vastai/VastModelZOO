# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
import time
from pathlib import Path
import shutil
import numpy as np
import yaml
import argparse
from easydict import EasyDict
from pcdet.datasets import KittiDataset


def get_fov_data(
        dataset_yaml,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        data_path=None,
        datatypes=None,  # 改为列表，支持多种数据类型
    ):
    
    if datatypes is None:
        datatypes = [np.float32]  # 默认只生成 float32
    
    dataset_cfg = EasyDict(yaml.safe_load(open(dataset_yaml)))
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=Path(data_path), training=False)
    dataset.set_split("val")
    
    # 创建每种数据类型的保存路径
    fov_paths = {}
    for dtype in datatypes:
        dtype_name = str(dtype).split(".")[-1].split("'")[0]
        fov_path = Path(data_path).joinpath('val', f'fov_pointcloud_{dtype_name}')
        fov_path.mkdir(parents=True, exist_ok=True)
        fov_paths[dtype] = fov_path
    
    # 创建其他目录
    calib_path = Path(data_path).joinpath("val", "calib")
    calib_path.mkdir(parents=True, exist_ok=True)
    
    label_2_path = Path(data_path).joinpath("val", "label_2")
    label_2_path.mkdir(parents=True, exist_ok=True)
    
    # 获取数据集信息
    kitti_infos = dataset.get_infos(num_workers=8, has_label=False, count_inside_pts=False)
    total_samples = len(dataset.sample_id_list)
    
    for index in range(total_samples):
        info = copy.deepcopy(kitti_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = dataset.get_calib(sample_idx)
        points = dataset.get_lidar(sample_idx)
        
        # 计算 FOV 标志
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        fov_flag = dataset.get_fov_flag(pts_rect, img_shape, calib)
        points_fov = points[fov_flag]
        
        # 为每种数据类型保存点云
        for dtype in datatypes:
            dtype_name = str(dtype).split(".")[-1].split("'")[0]
            output_file = fov_paths[dtype].joinpath(f'{sample_idx}.bin')
            
            # 转换为指定数据类型并保存
            points_fov.astype(dtype).tofile(str(output_file))
        
        # 复制 label 和 calib 文件
        label_file = Path(data_path).joinpath("training", "label_2", f'{sample_idx}.txt')
        calib_file = Path(data_path).joinpath("training", "calib", f'{sample_idx}.txt')
        
        if label_file.exists():
            shutil.copyfile(label_file, label_2_path.joinpath(f'{sample_idx}.txt'))
        else:
            print(f"警告: 标签文件不存在 {label_file}")
            
        if calib_file.exists():
            shutil.copyfile(calib_file, calib_path.joinpath(f'{sample_idx}.txt'))
        else:
            print(f"警告: 标定文件不存在 {calib_file}")
        
        time.sleep(0.5)  # 可选的延迟，避免资源占用过高


def main():
    parser = argparse.ArgumentParser(description="生成 FOV 点云数据")
    parser.add_argument("--dataset_yaml", type=str, 
                       default="/patn/to/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml", 
                       help="数据集配置文件")
    parser.add_argument("--data_path", type=str, 
                       default="/patn/to/OpenPCDet/data/kitti", 
                       help="数据路径")
    parser.add_argument("--dtypes", type=str, nargs='+',
                       choices=['float32', 'float16'],
                       default=['float32', 'float16'],
                       help="生成的数据类型，支持多种格式，如: float32 float16")
    
    args = parser.parse_args()
    
    # 将字符串参数转换为 numpy 数据类型
    datatypes = []
    dtype_map = {
        'float32': np.float32,
        'float16': np.float16
    }
    
    for dtype_str in args.dtypes:
        if dtype_str in dtype_map:
            datatypes.append(dtype_map[dtype_str])
        else:
            print(f"警告: 不支持的数据类型 {dtype_str}，已跳过")
    
    if not datatypes:
        print("错误: 未指定有效的数据类型，使用默认 float32")
        datatypes = [np.float32]
    
    print(f"将生成以下数据类型的 FOV 点云: {[str(dt).split('.')[-1] for dt in datatypes]}")
    
    get_fov_data(
        dataset_yaml=args.dataset_yaml,
        data_path=args.data_path,
        datatypes=datatypes
    )


if __name__ == '__main__':
    main()