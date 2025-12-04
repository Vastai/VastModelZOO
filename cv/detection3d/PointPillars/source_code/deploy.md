# Build_In Deploy

## step.1 安装OpenPCDet环境

- 安装OpenPCDet环境，需有CUDA环境机器，具体参考官方安装步骤：[docs/INSTALL.md](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md)
    ```bash
    git clone https://github.com/open-mmlab/OpenPCDet.git
    cd OpenPCDet
    git checkout 8cacccec11db6f59bf6934600c9a175dae254806

    conda create -n openpcdet python=3.10
    conda activate openpcdet
    # pip3 install torch torchvision
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

    pip install spconv-cu124
    python setup.py develop

    # https://github.com/open-mmlab/OpenPCDet/issues/1583
    pip install setuptools==58 av2 kornia==0.5.8 onnxsim
    pip install -r requirements.txt
    ```
## step.2 源码修改和onnx导出
> 在`OpenPCDet环境`下进行

1. 更改[vfe](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_3d/vfe/pillar_vfe.py#L94)输出, 参考[pillar_vfe](./onnx_utils/pillar_vfe.py)

   ```python
    class PillarVFE(VFETemplate):
        def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
            super().__init__(model_cfg=model_cfg)

            self.use_norm = self.model_cfg.USE_NORM
            self.with_distance = self.model_cfg.WITH_DISTANCE
            self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
            num_point_features += 6 if self.use_absolute_xyz else 3
            if self.with_distance:
                num_point_features += 1

            self.num_filters = self.model_cfg.NUM_FILTERS
            assert len(self.num_filters) > 0
            num_filters = [num_point_features] + list(self.num_filters)

            pfn_layers = []
            for i in range(len(num_filters) - 1):
                in_filters = num_filters[i]
                out_filters = num_filters[i + 1]
                pfn_layers.append(
                    PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
                )
            self.pfn_layers = nn.ModuleList(pfn_layers)

            self.voxel_x = voxel_size[0]
            self.voxel_y = voxel_size[1]
            self.voxel_z = voxel_size[2]
            self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
            self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
            self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        def get_output_feature_dim(self):
            return self.num_filters[-1]

        def get_paddings_indicator(self, actual_num, max_num, axis=0):
            actual_num = torch.unsqueeze(actual_num, axis + 1)
            max_num_shape = [1] * len(actual_num.shape)
            max_num_shape[axis + 1] = -1
            max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)  # 得到体素真实点max
            paddings_indicator = actual_num.int() > max_num # 得到各个体素中每个像素的mask,若为True则为真实坐标，若为False，则为补齐坐标
            return paddings_indicator   # [N,32]

        def forward(self, batch_dict, **kwargs):

            features = batch_dict['voxels']

            for pfn in self.pfn_layers:
                features = pfn(features)
            features = features.squeeze()
            batch_dict['pillar_features'] = features

            return batch_dict
   ```
2. 增加[PointPillarScatterFunction](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py)算子, 参考[pointpillar_scatter.py](./onnx_utils/pointpillar_scatter.py)

   ```python
   import torch
   import torch.nn as nn

   from torch.autograd import Function
   # 定义 map_roi_levels 反算金字塔层函数
   class PointPillarScatterFunction(Function):
       @staticmethod
       def forward(ctx, pillar_features, coords, mask, size_x, size_y, size_z, features):
           ctx.size_x = size_x
           ctx.size_y = size_y
           ctx.size_z = size_z
           ctx.features = features
           ctx.mask = mask

           mask = int(torch.sum(mask))
           print("mask is:",mask)
           coords = coords[:mask,:]
           pillar_features = pillar_features[:mask,...]
           batch_spatial_features = []
           batch_size = coords[:, 0].max().int().item() + 1
           for batch_idx in range(batch_size):
               spatial_feature = torch.zeros(
                   ctx.features,
                   ctx.size_x * ctx.size_y * ctx.size_z,
                   dtype=pillar_features.dtype,
                   device=pillar_features.device)

               batch_mask = coords[:, 0] == batch_idx
               this_coords = coords[batch_mask, :]
               indices = this_coords[:, 1] + this_coords[:, 2] * ctx.size_x + this_coords[:, 3]       # x,y转index z + y*W + x
               indices = indices.type(torch.long)
               pillars = pillar_features[batch_mask, :]
               pillars = pillars.t()   #转置，C,index
               spatial_feature[:, indices] = pillars
               batch_spatial_features.append(spatial_feature)

           batch_spatial_features = torch.stack(batch_spatial_features, 0)
           batch_spatial_features = batch_spatial_features.view(batch_size, ctx.features * ctx.size_z, ctx.size_y, ctx.size_x) # resize得到n,c,h,w的特征图
           return batch_spatial_features
       @staticmethod
       def symbolic(g,
                   *inputs):
           return g.op('vacc::PointPillarScatterFunction',
                       inputs[0],
                       inputs[1],
                       inputs[2],sizex_i = inputs[3], sizey_i = inputs[4], sizez_i = inputs[5], features_i = inputs[6])

   ppscatter_function = PointPillarScatterFunction.apply

   class PointPillarScatter(nn.Module):
       def __init__(self, model_cfg, grid_size, **kwargs):
           super().__init__()

           self.model_cfg = model_cfg
           self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
           self.nx, self.ny, self.nz = grid_size
           assert self.nz == 1

       def forward(self, batch_dict, **kwargs):
           pillar_features, coords, mask = batch_dict['pillar_features'], batch_dict['voxel_coords'], batch_dict['mask']
           batch_spatial_features = ppscatter_function(pillar_features,
                                                       coords,
                                                       mask,
                                                       self.nx,
                                                       self.ny,
                                                       self.nz,
                                                       self.num_bev_features)
           #[1,64,496,432]
           batch_dict['spatial_features'] = batch_spatial_features
           return batch_dict
   ```

3. onnx graph 算子注册，参考[`export.py`](./onnx_utils/export.py)

   > 导出前注册
   >

   ```python
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx.symbolic_helper import parse_args
    @parse_args("v","v","v","i","i","i","i")
    def symbolic(g, pillar_features,
                    coords,
                    mask,
                    size_x,size_y,size_z,
                    size_features):
        return g.op('custom::PointPillarScatterFunction',
                    pillar_features,
                    coords,
                    mask,
                    size_x,size_y,size_z,
                    size_features)
    register_custom_op_symbolic("vacc::PointPillarScatterFunction", symbolic, 1)
   ```

4. 修改模型返回值，PointPillar共有5个子模块，其中前4个为网络BackBone相关，因此只需要计算前4个即可

   1. 修改[pointpillar.py](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/detectors/pointpillar.py#L9), 参考[pointpillar.py](./onnx_utils/pointpillar.py)

      ```python
      def forward(self, batch_dict):
          for cur_module in self.module_list[0:4]:
              batch_dict = cur_module(batch_dict)
          return batch_dict['cls_preds'],batch_dict['box_preds'],batch_dict['dir_cls_preds']
      ```
   2. 修改[detection_head](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/dense_heads/anchor_head_single.py#L58) 返回值, 参考[anchor_head_single.py](./onnx_utils/anchor_head_single.py)

      ```python
      def forward(self, data_dict):
          spatial_features_2d = data_dict['spatial_features_2d']

          cls_preds = self.conv_cls(spatial_features_2d)
          box_preds = self.conv_box(spatial_features_2d)

          cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
          box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

          self.forward_ret_dict['cls_preds'] = cls_preds
          self.forward_ret_dict['box_preds'] = box_preds

          if self.conv_dir_cls is not None:
              dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
              dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
              self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
          else:
              dir_cls_preds = None

          # if output
          data_dict['cls_preds'] = cls_preds
          data_dict['box_preds'] = box_preds
          data_dict['dir_cls_preds'] = dir_cls_preds
          return data_dict
      ```

5. 执行onnx模型导出
    
    ckpt下载[pointpillar_7728.pth](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view)

   ```bash
    mv ./onnx_utils/export.py  /path/to/OpenPCDet/tools/ && cd /path/to/OpenPCDet/tools/
    python export.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml  --ckpt /path/to/pointpillar_7728.pth
   ```

## step.3 准备数据集

1. 获取KITTI数据集：[KITTI 3D object detection dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

   所需数据如下：

   - 分别是彩色图像数据（12GB）， data_object_image_2.zip
   - 点云数据（29GB），data_object_velodyne.zip
   - 相机矫正数据（16MB），data_object_calib.zip
   - 标签数据（5MB），data_object_label_2.zip

   其中彩色图像数据、点云数据、相机矫正数据均包含training（7481）和testing（7518）两个部分，标签数据只有training数据。

   ```bash
   mv data*.zip /path/to/OpenPCDet/data/kitti/ && cd /path/to/OpenPCDet/data/kitti/
   unzip xxx.zip
   ```

   ```bash
   OpenPCDet
   ├── data
   │   ├── kitti
   │   │   │── ImageSets
   │   │   │── training
   │   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
   │   │   │── testing
   │   │   │   ├──calib & velodyne & image_2
   ├── pcdet
   ├── tools
   ```

2. 下载数据划分文件

   ```bash
   # 下载数据划分文件
   # val.txt/test.txt/train.txt 原仓库已存在
   # https://github.com/open-mmlab/OpenPCDet/tree/master/data/kitti/ImageSets
   cd /path/to/OpenPCDet
   wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/trainval.txt
   ```

3. 划分数据集

   ```bash
   cd /path/to/OpenPCDet
   python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
   ```
   ```bash
   OpenPCDet
   ├── data
   │   ├── kitti
   │   │   │── gt_database
   │   │   │── ImageSets
   │   │   │── testing
   │   │   │── training
   │   │   │── kitti_dbinfos_train.pkl
   │   │   │── kitti_infos_test.pkl
   │   │   │── kitti_infos_train.pkl
   │   │   │── kitti_infos_trainval.pkl
   │   │   │── kitti_infos_val.pkl
   ```

4. 在kitty数据集官方benchmark数据结果的跑测中，是只用了前向数据的。在OpenPCDet官方工程中对Kitty数据的filter处理在 `pcdet/datasets/kitti_dataset.py`。本次数据预处理主要是从原始数据中提取并处理视野(Field of View, FOV)内的点云数据。参考[get_fov_data.py](../../common/kitti_eval/get_fov_data.py)

   ```bash
   cd ../../common/kitti_eval/
   python get_fov_data.py \
   --dataset_yaml /path/to/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml \
   --data_path /path/to/OpenPCDet/data/kitti
   ```
   ```bash
   #主要生成了val文件
   OpenPCDet
   ├── data
   │   ├── kitti
   │   │   │── val
   │   │   │   ├── calib & label_2
   │   │   │   ├── fov_pointcloud_float32
   │   │   │   ├── fov_pointcloud_float16
   ```

5. 校准数据集数据处理
   将点云数据体素化并保存为 `npz`文件,参考[pre_process.py](../../common/kitti_eval/pre_process.py)

   ```bash
   cd ../../common/kitti_eval/
   python pre_process.py \
   --kitti_data /path/to/fov_pointcloud_float32 \
   --save_path /path/to/fov_pointcloud_float32_npz 
   ```
## step.4 模型转换

1. `max_voxels_cur_cloud` 需要根据数据集设置为该数据集最大的 `voxel_numbers`, 此值会影响推理性能
2. 模型输入shape和onnx有区别，应设置为：

   ```yaml
   voxels: [max_voxels_cur_cloud, 32, 10]
   voxel_coords: [3, max_voxels_cur_cloud]
   mask: [1, max_voxels_cur_cloud]
   ```
3. 根据具体模型修改配置文件
    - 由于硬件的限制，当前只支持int8量化模型，不支持fp16量化模型
    - [cfg.yaml](../build_in/build/cfg.yaml)

    > - 编译参数`backend.type: tvm_vacc`
    > - int8精度: 编译参数`backend.dtype: int8`
    

4. 模型编译
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    cd PointPillars
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/cfg.yaml
    ```

## step.5 模型推理&评估

1. 参考[pointpillar_infer.py](../build_in/vsx/pointpillar_infer.py)生成推理的txt结果
    ```bash
    python3  ../build_in/vsx/pointpillar_infer.py \
        -m "[/path/to/pointpillar_int8/mod]" \
        --elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
        --max_voxel_num [16000] \
        --voxel_size [0.16,0.16,4] \
        --coors_range [0,-39.68,-3,69.12,39.68,1] \
        --feat_size [864,496,480,480] \
        --device_id 0 \
        --shuffle_enabled 0 \
        --normalize_enabled 0 \
        --max_points_num 12000000 \
        --dataset_root  /path/to/fov_pointcloud_float16 \
        --dataset_output_folder infer_output
    ```
    - 注意替换命令行中/path/to为实际路径

2. 精度评估
    - 使用[evaluation.py](../../common/kitti_eval/eval/evaluation.py)脚本进行精度评估
        - 精度评估命令为：
        ```
        python ../../common/kitti_eval/eval/evaluation.py --out_dir infer_output
        ```
        
## step.6 模型推理性能评估
1. 测试最大吞吐
    - 参考[pointpillar_prof.py](../build_in/vsx/pointpillar_prof.py)，测试最大吞吐
    ```bash
    python3 ../build_in/vsx/pointpillar_prof.py \
        -m "[/path/to/pointpillar_int8/mod]" \
        --elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
        --max_voxel_num [16000] \
        --max_points_num 12000000 \
        --voxel_size [0.16,0.16,4] \
        --coors_range [0,-39.68,-3,69.12,39.68,1] \
        --shuffle_enabled 0 \
        --normalize_enabled 0 \
        --feat_size [864,496,480,480] \
        --device_ids [0] \
        --shape [40000] \
        --batch_size 1 \
        --instance 1 \
        --iterations 1500 \
        --input_host 1 \
        --queue_size 1
    ```

2. 测试最小时延
    - 参考[pointpillar_prof.py](../build_in/vsx/pointpillar_prof.py)，测试最小时延
    ```bash
    python3 ../build_in/vsx/pointpillar_prof.py \
        -m "[/path/to/pointpillar_int8/mod]" \
        --elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
        --max_voxel_num [16000] \
        --max_points_num 12000000 \
        --voxel_size [0.16,0.16,4] \
        --coors_range [0,-39.68,-3,69.12,39.68,1] \
        --shuffle_enabled 0 \
        --normalize_enabled 0 \
        --feat_size [864,496,480,480] \
        --device_ids [0] \
        --shape [40000] \
        --batch_size 1 \
        --instance 1 \
        --iterations 1000 \
        --input_host 1 \
        --queue_size 0
    ```

