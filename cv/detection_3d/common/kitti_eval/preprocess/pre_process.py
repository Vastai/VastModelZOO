import glob
import os

import numpy as np

tv = None
try:
    import cumm.tensorview as tv
except:
    pass

# 1. mask_points_and_boxes_outside_range
def get_pointcloud_mask_by_limit_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
        & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


'''
spconv 点云转换器
'''

# 2. VoxelGeneratorV2
def create_voxel_generator(voxel_space_size,
                           coord_range,
                           num_point_features,
                           max_points_per_voxel,
                           max_voxels_per_cloud):

    try:
        from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
        spconv_ver = 1
    except:
        try:
            from spconv.utils import VoxelGenerator
            spconv_ver = 1
        except:
            from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
            spconv_ver = 2

    if spconv_ver == 1:
        voxel_generator = VoxelGenerator(
            voxel_size=voxel_space_size,
            point_cloud_range=coord_range,
            max_num_points=max_points_per_voxel,
            max_voxels=max_voxels_per_cloud
        )
    else:
        voxel_generator = VoxelGenerator(
            vsize_xyz=voxel_space_size,
            coors_range_xyz=coord_range,
            num_point_features=num_point_features,
            max_num_points_per_voxel=max_points_per_voxel,
            max_num_voxels=max_voxels_per_cloud
        )
    return voxel_generator


def get_paddings_indicator(actual_num, max_num, axis=0):
    actual_num = np.expand_dims(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = np.arange(max_num, dtype=np.int32).reshape(
        max_num_shape)  # 得到体素真实点max
    # 得到各个体素中每个像素的mask,若为True则为真实坐标，若为False，则为补齐坐标
    paddings_indicator = actual_num > max_num
    return paddings_indicator   # [N,32]


def point2voxel(points,
                max_voxels_cur_cloud,
                max_voxels_per_cloud,
                max_points_per_voxel,
                expand_batch,
                use_abslote_xyz,
                point_cloud_range_startx,
                point_cloud_range_starty,
                point_cloud_range_startz,
                point_cloud_range_endx,
                point_cloud_range_endy,
                point_cloud_range_endz,
                voxel_space_size_x,
                voxel_space_size_y,
                voxel_space_size_z,
                num_point_features = 4):

    avaliable_points = points

    masks = get_pointcloud_mask_by_limit_range(avaliable_points, [point_cloud_range_startx, point_cloud_range_starty, point_cloud_range_startz,
                                                                  point_cloud_range_endx,  point_cloud_range_endy,  point_cloud_range_endz])
    avaliable_points = avaliable_points[masks]

    voxel_generator = create_voxel_generator([voxel_space_size_x, voxel_space_size_y, voxel_space_size_z],
                                             [point_cloud_range_startx, point_cloud_range_starty, point_cloud_range_startz,
                                              point_cloud_range_endx,  point_cloud_range_endy,  point_cloud_range_endz],
                                             num_point_features,
                                             max_points_per_voxel,
                                             max_voxels_per_cloud)
    assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
    voxel_output = voxel_generator.point_to_voxel(
        tv.from_numpy(avaliable_points))

    tv_voxels, tv_coordinates, tv_num_points = voxel_output
    # 3. 将点云转换为voxel
    voxel_features = tv_voxels.numpy()
    coords = tv_coordinates.numpy()
    voxel_num_points = tv_num_points.numpy()

    # 4. 保留原来的维度信息&清楚补0的数据
    points_mean = voxel_features[:, :, :3].sum(
        axis=1, keepdims=True) / voxel_num_points.reshape(-1, 1, 1)
    f_cluster = voxel_features[:, :, :3] - points_mean  # 得偏移体素

    voxel_x = voxel_space_size_x
    voxel_y = voxel_space_size_y
    voxel_z = voxel_space_size_z

    x_offset = voxel_x / 2 + point_cloud_range_startx
    y_offset = voxel_y / 2 + point_cloud_range_starty
    z_offset = voxel_z / 2 + point_cloud_range_startz
    f_center = np.zeros_like(voxel_features[:, :, :3])
    # print("coord shape is:", coords.shape)
    # coord * voxel_x + x_offset = 体素单位的物理中心位置 体素单位的中心位置的偏移
    f_center[:, :, 0] = voxel_features[:, :, 0] - \
        (np.expand_dims(coords[:, 2], 1) * voxel_x + x_offset)
    f_center[:, :, 1] = voxel_features[:, :, 1] - \
        (np.expand_dims(coords[:, 1], 1) * voxel_y + y_offset)
    f_center[:, :, 2] = voxel_features[:, :, 2] - \
        (np.expand_dims(coords[:, 0], 1) * voxel_z + z_offset)

    if use_abslote_xyz:
        features = [voxel_features, f_cluster, f_center]
    else:
        features = [voxel_features[..., 3:], f_cluster, f_center]
    features = np.concatenate(features, axis=-1)

    # print(features.shape)
    voxel_count = features.shape[1]
    mask = get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
    mask = np.expand_dims(mask, -1)
    features *= mask

    # 5. 维度转换
    coords = coords[:, :]

    real_counts = coords.shape[0]
    out_mask = np.zeros((max_voxels_cur_cloud), dtype=np.int8)
    out_mask[:real_counts] = 1

    out_coords = np.zeros((3, max_voxels_cur_cloud), dtype=np.int16)
    out_coords[:, :real_counts] = coords.transpose(1, 0)

    # print("out coord shape is:", out_coords.shape)

    input_channel = 3 if use_abslote_xyz else points.shape[1]
    expand_channel = 7 if expand_batch else 6
    out_channel = input_channel + expand_channel

    out_features = np.zeros(
        (max_voxels_cur_cloud, max_points_per_voxel, out_channel), dtype=np.float32)
    out_features[:real_counts, :, :] = features

    return out_features, out_coords, out_mask


if __name__ == '__main__':
    quant_data_path = "./projects/adas/OpenPCDet/data/kitti/val/fov_pointcloud_float32"
    save_path = "./projects/adas/OpenPCDet/data/kitti/val/fov_pointcloud_float32_npz"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset = []
    quant_data_all_file = f"{quant_data_path}/*.bin"
    data_list = glob.glob(quant_data_all_file, recursive=True)
    data_list.sort()
    all_files = [f for f in data_list]

    quant_count = len(all_files)
    for file in range(quant_count):
        sample_file = all_files[file]
        points = np.fromfile(sample_file, dtype=np.float32).reshape(-1, 4)
        features, coords, masks = point2voxel(points,
                                              max_voxels_cur_cloud=16000,
                                              max_voxels_per_cloud=40000,
                                              max_points_per_voxel=32,
                                              expand_batch=True,
                                              use_abslote_xyz=True,
                                              point_cloud_range_startx=0,
                                              point_cloud_range_starty=-39.68,
                                              point_cloud_range_startz=-3,
                                              point_cloud_range_endx=69.12,
                                              point_cloud_range_endy=39.68,
                                              point_cloud_range_endz=1,
                                              voxel_space_size_x=0.16,
                                              voxel_space_size_y=0.16,
                                              voxel_space_size_z=4)
        features = features.transpose(2, 1, 0)
        features = np.expand_dims(features, 0)

        coords = coords[1:, :]
        coords = np.pad(coords, ((0, 1), (0, 0)))
        coords2 = coords.copy()
        coords2[0, :] = coords[1, :]
        coords2[1, :] = coords[0, :]
        coords = coords2

        print("features:",features.shape,features.dtype, "coords:",coords.shape,coords.dtype, "mask:", masks.shape,masks.dtype)
        np.savez(
                os.path.join(save_path, os.path.splitext(os.path.split(sample_file)[-1])[0] + ".npz"),
                **{
                    "voxels" : features,
                    "voxel_coords" : coords,
                    "mask" : masks,
                }
            )
