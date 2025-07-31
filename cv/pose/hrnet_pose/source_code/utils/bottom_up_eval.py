# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import numpy as np
import torch
import math
# from mmpose.core.post_processing import (get_warp_matrix, transform_preds,
#                                          warp_affine_joints)


def split_ae_outputs(outputs, num_joints, with_heatmaps, with_ae,
                     select_output_index):
    """Split multi-stage outputs into heatmaps & tags.

    Args:
        outputs (list(Tensor)): Outputs of network
        num_joints (int): Number of joints
        with_heatmaps (list[bool]): Option to output
            heatmaps for different stages.
        with_ae (list[bool]): Option to output
            ae tags for different stages.
        select_output_index (list[int]): Output keep the selected index

    Returns:
        tuple: A tuple containing multi-stage outputs.

        - list[Tensor]: multi-stage heatmaps.
        - list[Tensor]: multi-stage tags.
    """

    heatmaps = []
    tags = []

    # aggregate heatmaps from different stages
    for i, output in enumerate(outputs):
        if i not in select_output_index:
            continue
        # staring index of the associative embeddings
        offset_feat = num_joints if with_heatmaps[i] else 0
        if with_heatmaps[i]:
            heatmaps.append(output[:, :num_joints])
        if with_ae[i]:
            tags.append(output[:, offset_feat:])
    
    return heatmaps, tags

def get_warp_matrix(theta, size_input, size_dst, size_target):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        np.ndarray: A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
                              0.5 * size_input[1] * math.sin(theta) +
                              0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
                              0.5 * size_input[1] * math.cos(theta) +
                              0.5 * size_target[1])
    return matrix

def flip_feature_maps(feature_maps, flip_index=None):
    """Flip the feature maps and swap the channels.

    Args:
        feature_maps (list[Tensor]): Feature maps.
        flip_index (list[int] | None): Channel-flip indexes.
            If None, do not flip channels.

    Returns:
        list[Tensor]: Flipped feature_maps.
    """
    flipped_feature_maps = []
    for feature_map in feature_maps:
        feature_map = torch.flip(feature_map, [3])
        if flip_index is not None:
            flipped_feature_maps.append(feature_map[:, flip_index, :, :])
        else:
            flipped_feature_maps.append(feature_map)

    return flipped_feature_maps


def _resize_average(feature_maps, align_corners, index=-1, resize_size=None):
    """Resize the feature maps and compute the average.

    Args:
        feature_maps (list[Tensor]): Feature maps.
        align_corners (bool): Align corners when performing interpolation.
        index (int): Only used when `resize_size' is None.
            If `resize_size' is None, the target size is the size
            of the indexed feature maps.
        resize_size (list[int, int]): The target size [h, w].

    Returns:
        list[Tensor]: Averaged feature_maps.
    """

    if feature_maps is None:
        return None
    feature_maps_avg = 0

    feature_map_list = _resize_concate(
        feature_maps, align_corners, index=index, resize_size=resize_size)
    for feature_map in feature_map_list:
        feature_maps_avg += feature_map

    feature_maps_avg /= len(feature_map_list)
    return [feature_maps_avg]


def _resize_unsqueeze_concat(feature_maps,
                             align_corners,
                             index=-1,
                             resize_size=None):
    """Resize, unsqueeze and concatenate the feature_maps.

    Args:
        feature_maps (list[Tensor]): Feature maps.
        align_corners (bool): Align corners when performing interpolation.
        index (int): Only used when `resize_size' is None.
            If `resize_size' is None, the target size is the size
            of the indexed feature maps.
        resize_size (list[int, int]): The target size [h, w].

    Returns:
        list[Tensor]: Averaged feature_maps.
    """
    if feature_maps is None:
        return None
    feature_map_list = _resize_concate(
        feature_maps, align_corners, index=index, resize_size=resize_size)

    feat_dim = len(feature_map_list[0].shape) - 1
    output_feature_maps = torch.cat(
        [torch.unsqueeze(fmap, dim=feat_dim + 1) for fmap in feature_map_list],
        dim=feat_dim + 1)
    return [output_feature_maps]


def _resize_concate(feature_maps, align_corners, index=-1, resize_size=None):
    """Resize and concatenate the feature_maps.

    Args:
        feature_maps (list[Tensor]): Feature maps.
        align_corners (bool): Align corners when performing interpolation.
        index (int): Only used when `resize_size' is None.
            If `resize_size' is None, the target size is the size
            of the indexed feature maps.
        resize_size (list[int, int]): The target size [h, w].

    Returns:
        list[Tensor]: Averaged feature_maps.
    """
    if feature_maps is None:
        return None

    feature_map_list = []

    if index < 0:
        index += len(feature_maps)

    if resize_size is None:
        resize_size = (feature_maps[index].size(2),
                       feature_maps[index].size(3))

    for feature_map in feature_maps:
        ori_size = (feature_map.size(2), feature_map.size(3))
        if ori_size != resize_size:
            feature_map = torch.nn.functional.interpolate(
                feature_map,
                size=resize_size,
                mode='bilinear',
                align_corners=align_corners)

        feature_map_list.append(feature_map)

    return feature_map_list


def aggregate_stage_flip(feature_maps,
                         feature_maps_flip,
                         index=-1,
                         project2image=True,
                         size_projected=None,
                         align_corners=False,
                         aggregate_stage='concat',
                         aggregate_flip='average'):
    """Inference the model to get multi-stage outputs (heatmaps & tags), and
    resize them to base sizes.

    Args:
        feature_maps (list[Tensor]): feature_maps can be heatmaps,
            tags, and pafs.
        feature_maps_flip (list[Tensor] | None): flipped feature_maps.
            feature maps can be heatmaps, tags, and pafs.
        project2image (bool): Option to resize to base scale.
        size_projected (list[int, int]): Base size of heatmaps [w, h].
        align_corners (bool): Align corners when performing interpolation.
        aggregate_stage (str): Methods to aggregate multi-stage feature maps.
            Options: 'concat', 'average'. Default: 'concat.

            - 'concat': Concatenate the original and the flipped feature maps.
            - 'average': Get the average of the original and the flipped
                feature maps.
        aggregate_flip (str): Methods to aggregate the original and
            the flipped feature maps. Options: 'concat', 'average', 'none'.
            Default: 'average.

            - 'concat': Concatenate the original and the flipped feature maps.
            - 'average': Get the average of the original and the flipped
                feature maps..
            - 'none': no flipped feature maps.

    Returns:
        list[Tensor]: Aggregated feature maps with shape [NxKxWxH].
    """

    if feature_maps_flip is None:
        aggregate_flip = 'none'

    output_feature_maps = []

    if aggregate_stage == 'average':
        _aggregate_stage_func = _resize_average
    elif aggregate_stage == 'concat':
        _aggregate_stage_func = _resize_concate
    else:
        NotImplementedError()

    if project2image and size_projected:
        _origin = _aggregate_stage_func(
            feature_maps,
            align_corners,
            index=index,
            resize_size=(size_projected[1], size_projected[0]))

        _flipped = _aggregate_stage_func(
            feature_maps_flip,
            align_corners,
            index=index,
            resize_size=(size_projected[1], size_projected[0]))
    else:
        _origin = _aggregate_stage_func(
            feature_maps, align_corners, index=index, resize_size=None)
        _flipped = _aggregate_stage_func(
            feature_maps_flip, align_corners, index=index, resize_size=None)

    if aggregate_flip == 'average':
        assert feature_maps_flip is not None
        for _ori, _fli in zip(_origin, _flipped):
            output_feature_maps.append((_ori + _fli) / 2.0)

    elif aggregate_flip == 'concat':
        assert feature_maps_flip is not None
        output_feature_maps.append(*_origin)
        output_feature_maps.append(*_flipped)

    elif aggregate_flip == 'none':
        if isinstance(_origin, list):
            output_feature_maps.append(*_origin)
        else:
            output_feature_maps.append(_origin)
    else:
        NotImplementedError()

    return output_feature_maps


def aggregate_scale(feature_maps_list,
                    align_corners=False,
                    project2image=True,
                    size_projected=None,
                    aggregate_scale='average'):
    """Aggregate multi-scale outputs.

    Note:
        batch size: N
        keypoints num : K
        heatmap width: W
        heatmap height: H

    Args:
        feature_maps_list (list[Tensor]): Aggregated feature maps.
        project2image (bool): Option to resize to base scale.
        size_projected (list[int, int]): Base size of heatmaps [w, h].
        align_corners (bool): Align corners when performing interpolation.
        aggregate_scale (str): Methods to aggregate multi-scale feature maps.
            Options: 'average', 'unsqueeze_concat'.

            - 'average': Get the average of the feature maps.
            - 'unsqueeze_concat': Concatenate the feature maps along new axis.
                Default: 'average.

    Returns:
        Tensor: Aggregated feature maps.
    """

    resize_size = None
    if project2image and size_projected:
        resize_size = (size_projected[1], size_projected[0])

    if aggregate_scale == 'average':
        output_feature_maps = _resize_average(
            feature_maps_list, align_corners, index=0, resize_size=resize_size)

    elif aggregate_scale == 'unsqueeze_concat':
        output_feature_maps = _resize_unsqueeze_concat(
            feature_maps_list, align_corners, index=0, resize_size=resize_size)
    else:
        NotImplementedError()

    return output_feature_maps[0]


def get_group_preds(grouped_joints,
                    center,
                    scale,
                    heatmap_size,
                    use_udp=False):
    """Transform the grouped joints back to the image.

    Args:
        grouped_joints (list): Grouped person joints.
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        heatmap_size (np.ndarray[2, ]): Size of the destination heatmaps.
        use_udp (bool): Unbiased data processing.
             Paper ref: Huang et al. The Devil is in the Details: Delving into
             Unbiased Data Processing for Human Pose Estimation (CVPR'2020).

    Returns:
        list: List of the pose result for each person.
    """
    if len(grouped_joints) == 0:
        return []

    if use_udp:
        if grouped_joints[0].shape[0] > 0:
            heatmap_size_t = np.array(heatmap_size, dtype=np.float32) - 1.0
            trans = get_warp_matrix(
                theta=0,
                size_input=heatmap_size_t,
                size_dst=scale,
                size_target=heatmap_size_t)
            grouped_joints[0][..., :2] = \
                warp_affine_joints(grouped_joints[0][..., :2], trans)
        results = [person for person in grouped_joints[0]]
    else:
        results = []
        for person in grouped_joints[0]:
            joints = transform_preds(person, center, scale, heatmap_size)
            results.append(joints)

    return results

def warp_affine_joints(joints, mat):
    """Apply affine transformation defined by the transform matrix on the
    joints.

    Args:
        joints (np.ndarray[..., 2]): Origin coordinate of joints.
        mat (np.ndarray[3, 2]): The affine matrix.

    Returns:
        np.ndarray[..., 2]: Result coordinate of joints.
    """
    joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return np.dot(
        np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1),
        mat.T).reshape(shape)

def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = coords.copy()
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords
