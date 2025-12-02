import argparse
import glob
from pathlib import Path

# # try:
# import mayavi.mlab as mlab
# from visual_utils import visualize_utils as V

# OPEN3D_FLAG = False
# # except:
# # import open3d
# # from visual_utils import open3d_vis_utils as V

# # OPEN3D_FLAG = True

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from onnxsim import simplify
import onnx

class DemoDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None,
        ext='.bin',
    ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = (
            glob.glob(str(root_path / f'*{self.ext}'))
            if self.root_path.is_dir()
            else [self.root_path]
        )

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            print("self.sample_file_list[index]: {name}".format(name=self.sample_file_list[index]))
            id_num = int(self.sample_file_list[index].split('/')[-1].split('.')[0].split('_')[-1])
            if (id_num == 1200):
                print()
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
            # xjg
            # Check if intensity normalization is required
            intensity = points[:, -1]
            if intensity.max() > 1.0 or intensity.min() < 0.0:
                # Normalize intensity to the range [0, 1]
                points[:, -1] = (intensity - intensity.min()) / (intensity.max() - intensity.min())
            print()
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_int_list(value):
    return [int(x) for x in value.split(',')]

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument(
        '--cfg_file',
        type=str,
        default='/path/to/model_centerpoint_pp_stride222.yaml',
        help='specify the config for demo',
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='/path/to/checkpoint_epoch_120.pth',
        help='specify the pretrained model',
    )
    parser.add_argument(
        '--max_voxel_num',
        type=int,
        default='32000',
        help='max voxel num',
    )
    parser.add_argument(
        "--backbone_input_shape",
        type=parse_int_list,
        default = [1,64,480,480], help = "backbone input shape"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default='./center_point_zte_v5',
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default='./in_data/', 
        help='specify the point cloud data file or directory',
    )
    parser.add_argument(
        '--ext',
        type=str,
        default='.bin',
        help='specify the extension of your point cloud data file',
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    logger = common_utils.create_logger()
    logger.info('-----------------Export ONNX---------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        ext=args.ext,
        logger=logger,
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    
    export_onnx = True  # 导出ONNX模型
    if export_onnx:
        # 导出ONNX模型
        cfg.MODEL['EXPORT_FLAG'] = 0
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
    
        dummy_input = torch.randn(args.max_voxel_num, 32, 10).cuda()  # 根据您的模型输入形状替换这行 6445 
        onnx_file_path = os.path.join(args.save_path, 'PillarVFE.onnx')  # 自定义ONNX输出路径
        torch.onnx.export(model, dummy_input, onnx_file_path, 
                export_params=True, 
                opset_version=11,  # 根据需要选择版本
                do_constant_folding=True,  # 启用常量折叠
                input_names=['input'],   # 输入名称
                output_names=['pillar_features'],  # 输出名称
                )
        
        path_onnx_sim = onnx_file_path.replace('.onnx', '_sim.onnx')
        model_onnx = onnx.load(onnx_file_path)
        model_simplified, check = simplify(model_onnx)
        assert check, "Simplified ONNX model could not be validated."
        onnx.save(model_simplified, path_onnx_sim)
        
        
        cfg.MODEL['EXPORT_FLAG'] = 1
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
        dummy_input = torch.randn(args.backbone_input_shape[0],args.backbone_input_shape[1],args.backbone_input_shape[2], args.backbone_input_shape[2]).cuda()  # 根据您的模型输入形状替换这行
        onnx_file_path = os.path.join(args.save_path, 'BaseBEVBackbone_CenterHead.onnx')  # 自定义ONNX输出路径
        torch.onnx.export(model, dummy_input, onnx_file_path, 
                        export_params=True, 
                        opset_version=11,  # 根据需要选择版本
                        do_constant_folding=True,  # 启用常量折叠
                        input_names=['spatial_features'],   # 输入名称
                        output_names=['center', 'center_z', 'dim', 'rot', 'hm'],  # 输出名称
                        ) 
        path_onnx_sim = onnx_file_path.replace('.onnx', '_sim.onnx')
        model_onnx = onnx.load(onnx_file_path)
        model_simplified, check = simplify(model_onnx)
        assert check, "Simplified ONNX model could not be validated."
        onnx.save(model_simplified, path_onnx_sim)
        
        print("EXported OK!!!")
        exit()
        
        # logger.info(f'ONNX model exported to {onnx_file_path}')

if __name__ == '__main__':
    main()
