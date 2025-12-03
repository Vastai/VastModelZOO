# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
import copy
import pickle
# from .ops import pp3dfunction
from .calibration_kitti import Calibration as calibObj
import os
work_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def generator_pointpillar_anchor(grid_size    = [432, 496, 1],
                                 anchor_range = [0.0, -39.68, -3.0, 
                                                 69.12, 39.68, 1.0], 
                                 grid_stride  = [2, 2, 2],
                                 anchor_sizes = [[[3.9, 1.6, 1.56]],
                                                 [[0.8, 0.6, 1.73]],
                                                 [[1.76, 0.6, 1.73]]],
                                 anchor_rotations = [[0, 1.57],
                                                     [0, 1.57],
                                                     [0, 1.57]],
                                 anchor_heights = [[-1.78], [-0.6], [-0.6]],
                                 align_center   = [False, False, False]):
        all_anchors = []
        num_anchors_per_location = []
        grid_sizes = [np.array(grid_size[:2]) // strides for strides in grid_stride]
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, anchor_sizes, anchor_rotations, anchor_heights, align_center):

            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            if align_center:
                x_stride = (anchor_range[3] - anchor_range[0]) / grid_size[0]
                y_stride = (anchor_range[4] - anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (anchor_range[3] - anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (anchor_range[4] - anchor_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0

            x_shifts = torch.arange(
                anchor_range[0] + x_offset, anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            )
            y_shifts = torch.arange(
                anchor_range[1] + y_offset, anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            )
            z_shifts = x_shifts.new_tensor(anchor_height)

            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid]
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x, y, z, 1, 3]
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)    # anchor_size 重复 size个，当前只有一个anchor，因此只有一份
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])  # 为每个特征点匹配一个anchor size
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1) # [216,248,1,1,2,6]
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # [x, y, z, num_size, num_rot, 7]

            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            #anchors = anchors.view(-1, anchors.shape[-1])
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors)
        return all_anchors, num_anchors_per_location

def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)

def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib):
    """
    :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param calib:
    :return:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    """
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    xyz_lidar = boxes3d_lidar_copy[:, 0:3]
    l, w, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    r = boxes3d_lidar_copy[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)
    # xyz_cam[:, 1] += h.reshape(-1) / 2
    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)

def boxes3d_kitti_camera_to_imageboxes(boxes3d, calib, image_shape=None):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    :param calib:
    :return:
        box_2d_preds: (N, 4) [x1, y1, x2, y2]
    """
    corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
    pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
    max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d_image

def generate_prediction_dicts(lidar_index, calib, img_shape, pred_dicts, class_names, output_path=None):
    """
    Args:
        batch_dict:
            frame_id:
        pred_dicts: list of pred_dicts
            pred_boxes: (N, 7), Tensor
            pred_scores: (N), Tensor
            pred_labels: (N), Tensor
        class_names:
        output_path:

    Returns:

    """
    def get_template_prediction(num_samples):
        ret_dict = {
            'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
            'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
            'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
            'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
            'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
        }
        return ret_dict

    def generate_single_sample_dict(batch_index, box_dict):
        pred_scores = box_dict['pred_scores']
        pred_boxes = box_dict['pred_boxes']
        pred_labels = box_dict['pred_labels']
        pred_dict = get_template_prediction(pred_scores.shape[0])
        if pred_scores.shape[0] == 0:
            return pred_dict

        image_shape = np.array(img_shape)
        pred_boxes_camera = boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
        pred_boxes_img = boxes3d_kitti_camera_to_imageboxes(
            pred_boxes_camera, calib, image_shape=image_shape
        )
        pred_dict['name'] = np.array(class_names)[pred_labels - 1]
        pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
        pred_dict['bbox'] = pred_boxes_img
        pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
        pred_dict['location'] = pred_boxes_camera[:, 0:3]
        pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
        pred_dict['score'] = pred_scores
        pred_dict['boxes_lidar'] = pred_boxes

        return pred_dict

    annos = []
    for index, box_dict in enumerate(pred_dicts):
        frame_id = lidar_index

        single_pred_dict = generate_single_sample_dict(index, box_dict)
        single_pred_dict['frame_id'] = frame_id
        annos.append(single_pred_dict)

        if output_path is not None:
            cur_det_file = output_path + ('/%s.txt' % frame_id)
            with open(cur_det_file, 'w') as f:
                bbox = single_pred_dict['bbox']
                loc = single_pred_dict['location']
                dims = single_pred_dict['dimensions']  # lhw -> hwl

                for idx in range(len(bbox)):
                    print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                            % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                single_pred_dict['score'][idx]), file=f)

    return annos

all_anchors_gen, num_anchors_list = generator_pointpillar_anchor()
val_infos = None
# kitti_infos_val.pkl存放了val数据集的point_cloud\image\image_shape相关的信息,calib:P2\R0_rect\Tr_velo_to_cam矩阵信息和对应的标签annos
with open(os.path.join(work_dir, "evals/kitti/kitti_infos_val.pkl"), 'rb') as f:
    val_infos = pickle.load(f)

def generater_results_by_output_with_detection(index, scores, labels, box,):
    global all_anchors_gen
    global num_anchors_list
    global val_infos
    
    # 获取index名称：点云index
    lidar_index = val_infos[index]['point_cloud']['lidar_idx']
    # 2dbox shape
    image_shape =  val_infos[index]['image']['image_shape']
    # 读取预测结果
    # 读取calib对象
    calib_path = work_dir + "/evals/kitti/calib/{0}.txt".format(lidar_index)
    calib = calibObj(calib_path)
    # generator anchors
    
    cls_preds = torch.from_numpy(scores)
    box_preds = torch.from_numpy(labels)
    dir_cls_preds = torch.from_numpy(box)  
    
    # return
    record_dict = {
        'pred_boxes': box,
        'pred_scores': scores,
        'pred_labels': labels
    }
    
    #return [record_dict], None
    generate_prediction_dicts(lidar_index, calib, image_shape, [record_dict], ["Car","Pedestrian","Cyclist"], work_dir + "/kitti_eval_system/results/data")
