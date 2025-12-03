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
from easydict import EasyDict
from pcdet.datasets import KittiDataset


def get_fov_data(
        dataset_yaml,
        class_names = ['Car', 'Pedestrian', 'Cyclist'],
        data_path = None,
        datatype = np.float16,
    ):

    dataset_cfg = EasyDict(yaml.safe_load(open(dataset_yaml)))
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=Path(data_path), training=False)
    dataset.set_split("val")
    kitti_infos = dataset.get_infos(num_workers=8, has_label=True, count_inside_pts=True)
    # get val pointcloud
    fov_pointcloud_path = Path(data_path).joinpath('val','fov_pointcloud_%s'% str(datatype).split(".")[-1].split("'")[0])
    # 保存路径
    Path(fov_pointcloud_path).mkdir(parents=True, exist_ok=True)

    calib_path = Path(data_path).joinpath("val","calib")
    Path(calib_path).mkdir(parents=True, exist_ok=True)

    label_2_path = Path(data_path).joinpath("val","label_2")
    Path(label_2_path).mkdir(parents=True, exist_ok=True)

    kitti_infos = dataset.get_infos(num_workers=8, has_label=False, count_inside_pts=False)
    for index in range(len(dataset.sample_id_list)):
        info = copy.deepcopy(kitti_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = dataset.get_calib(sample_idx)
        points = dataset.get_lidar(sample_idx)
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        fov_flag = dataset.get_fov_flag(pts_rect, img_shape, calib)
        points = points[fov_flag]
        points.astype(datatype).tofile("%s/%s.bin"%(fov_pointcloud_path, sample_idx))
        time.sleep(0.5)
        # get label & calib
        label_file = Path(data_path).joinpath("training","label_2",'%s.txt' % sample_idx)
        calib_file = Path(data_path).joinpath("training","calib",'%s.txt' % sample_idx)

        shutil.copyfile(label_file, label_2_path.joinpath('%s.txt' % sample_idx))
        shutil.copyfile(calib_file, calib_path.joinpath('%s.txt' % sample_idx))


if __name__ == '__main__':
    dataset_yaml = "./projects/adas/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml"
    data_path = "./projects/adas/OpenPCDet/data/kitti"
    datatype = np.float16

    get_fov_data(dataset_yaml, data_path = data_path, datatype = datatype)