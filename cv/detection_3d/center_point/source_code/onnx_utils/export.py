import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network
from pcdet.utils import common_utils
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args


@parse_args("v", "v", "v", "i", "i", "i", "i")
def symbolic(g, pillar_features, coords, mask, size_x, size_y, size_z, size_features):
    return g.op(
        'custom::PointPillarScatterFunction',
        pillar_features,
        coords,
        mask,
        size_x,
        size_y,
        size_z,
        size_features,
    )


register_custom_op_symbolic("vacc::PointPillarScatterFunction", symbolic, 11)


class DemoDataset(DatasetTemplate):
    def __init__(
        self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'
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
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
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


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument(
        '--cfg_file',
        type=str,
        default='./project/vehicle-road-collaboration-lance_det3d/model_centerpoint_pp.yaml',
        help='specify the config for demo',
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='./project/vehicle-road-collaboration-lance_det3d/Zhongxing_dense_crowd_v1/lidar_data_fp32_332.bin',
        help='specify the point cloud data file or directory',
    )
    parser.add_argument(
        '--ckpt', type=str, default="./project/vehicle-road-collaboration-lance_det3d/checkpoint_epoch_126.pth", help='specify the pretrained model'
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
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        ext=args.ext,
        logger=logger,
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    print("model is:", model)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)

    ############## onnx 导出################################
    np.set_printoptions(threshold=np.inf)
    with torch.no_grad():

        MAX_VOXELS = 32000

        dummy_voxels = torch.zeros((MAX_VOXELS, 32, 10), dtype=torch.float32, device='cpu')

        dummy_voxel_idxs = torch.zeros((MAX_VOXELS, 4), dtype=torch.int16, device='cpu')

        dummy_mask = torch.ones((1, MAX_VOXELS), dtype=torch.int8, device='cpu')

        dummy_input = dict()
        dummy_input['voxels'] = dummy_voxels
        dummy_input['voxel_coords'] = dummy_voxel_idxs
        dummy_input['mask'] = dummy_mask
        # dummy_input['spatial_features'] = dummy_spatial_features
        dummy_input['batch_size'] = 1

        torch.onnx.export(
            model,  # model being run
            dummy_input,  # model input (or a tuple for multiple inputs)
            "./project/vehicle-road-collaboration-lance_det3d/onnx_centerpoint/pointpillar_openpcdet_32000.onnx",  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=11,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            keep_initializers_as_inputs=True,
            # custom_opsets = 10
            input_names=['voxels', 'voxel_coords', 'mask'],  # the model's input names
            # input_names = ['spatial_features',],
            output_names=['cls_preds', 'box_preds', 'dir_cls_preds'],  # the model's output names, 置信度，box,方向分类(前或后)
        )
    print("export ok")
    exit(0)


if __name__ == '__main__':
    main()
