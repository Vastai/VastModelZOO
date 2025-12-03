import argparse
import os

import numpy as np
import torch
import yaml
from easydict import EasyDict
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils


def decode_vacc_result_npz(result_npz, frame_id_list, maks_score=0):
    pred_dicts = []
    for id in frame_id_list:
        data = np.load(os.path.join(result_npz, id + '.npz'))

        out = min(len(data['boxes']) // 7, len(data['score']))
        out = min(out, len(data['label']))
        if len(data['boxes'].shape) == 1:
            try:
                mask = data['score'][:out] >= maks_score
                scores = data['score'][:out][mask]
                labels = data['label'][:out][mask]
                box = data['boxes'][:out * 7].reshape(-1, 7)[mask]
            except:
                print('score shape  ', data['score'].shape)
                print('boxes shape  ', data['boxes'].shape)
            # box = data["boxes"].reshape(-1, 7)[mask]
        else:
            mask = data['score'] >= maks_score
            box = data['boxes'][mask]
            scores = data['score'][mask]
            labels = data['label'][mask]

        box = box.astype(np.float32)
        scores = scores.astype(np.float32)
        labels = labels.astype(np.int32)
        record_dict = {
            'pred_boxes': check_numpy_to_torch(box),
            'pred_scores': check_numpy_to_torch(scores),
            'pred_labels': check_numpy_to_torch(labels),
        }
        pred_dicts.append(record_dict)
    return pred_dicts


def check_numpy_to_torch(x):
    return torch.from_numpy(x)


def get_annos(dataset_yaml, class_names):
    logger = common_utils.create_logger()
    det_annos = []
    cfg = EasyDict(yaml.safe_load(open(dataset_yaml)))

    dataset, test_loader, _ = build_dataloader(
        dataset_cfg=cfg,
        # class_names=["car", "truck", "bus", "non_motor_vehicleslist", "pedestrians"],
        # class_names=['Car', 'Pedestrian', 'Cyclist'],
        class_names=class_names,
        batch_size=8,
        dist=False,
        workers=8,
        logger=logger,
        training=False,
    )

    class_names = dataset.class_names

    for i, batch_dict in enumerate(test_loader):
        pred_dicts = decode_vacc_result_npz(
            result_npz=args.result_npz, frame_id_list=batch_dict['frame_id'])
        annos = dataset.generate_prediction_dicts(batch_dict, pred_dicts,
                                                  class_names)

        det_annos += annos

    # ap
    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    print('Average predicted number of objects(%d samples): %.3f' %
          (len(det_annos), total_pred_objects / max(1, len(det_annos))))
    result_str, result_dict = dataset.evaluation(
        det_annos,
        class_names,
        point_cloud_range=cfg.POINT_CLOUD_RANGE,
        eval_metric='kitti')

    print(result_str)


def parse_list(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def get_argparse():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument(
        '--dataset_yaml',
        type=str,
        default='config/model_centerpoint_pp_stride222_dataset.yaml',
        help='specify the config of dataset',
    )
    parser.add_argument(
        '--result_npz',
        type=str,
        default='runstream_output',
        help='specify the path of result',
    )
    parser.add_argument('--class_names',
                        type=parse_list,
                        default=[
                            'car', 'truck', 'bus', 'non_motor_vehicleslist',
                            'pedestrians'
                        ],
                        help='class names')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    print(args)
    get_annos(dataset_yaml=args.dataset_yaml, class_names=args.class_names)
