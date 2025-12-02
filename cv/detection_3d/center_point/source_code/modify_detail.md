
## 模型修改和导出细节

### modify.1 修改PillarVFE类实现
- **注释**[OpenPCDet/pcdet/models/backbones_3d/vfe/pillar_vfe.py#L94](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_3d/vfe/pillar_vfe.py#L94)的forward函数，替换成如下内容
    - 建议直接使用此已修改文件：[pillar_vfe.py](../source_code/onnx_utils/pillar_vfe.py)，去替换原始同名文件
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
                    PFNLayer(
                        in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2)
                    )
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
            max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
                max_num_shape
            )
            paddings_indicator = actual_num.int() > max_num
            return paddings_indicator
        
        # export onnx
        def forward(self, features, **kwargs):
            for pfn in self.pfn_layers:
                features = pfn(features)
            features = features.squeeze()
            pillar_features = features
            return pillar_features

        # NOTE(lance)
        class PillarVFE2(VFETemplate):
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
                        PFNLayer(
                            in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2)
                        )
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
                max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
                    max_num_shape
                )  # 得到体素真实点max
                paddings_indicator = (
                    actual_num.int() > max_num
                )  # 得到各个体素中每个像素的mask,若为True则为真实坐标，若为False，则为补齐坐标
                return paddings_indicator  # [N,32]

            def forward(self, batch_dict, **kwargs):

                features = batch_dict['voxels']

                for pfn in self.pfn_layers:
                    features = pfn(features)
                features = features.squeeze()
                batch_dict['pillar_features'] = features

                return batch_dict
    ```

### modify.2 修改PointPillarScatter类实现
- 修改[PointPillarScatterFunction](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py)
    - 建议直接使用此已修改文件：[pointpillar_scatter.py](../source_code/onnx_utils/pointpillar_scatter.py)，去替换原始同名文件

    ```python
    import torch
    import torch.nn as nn


    class PointPillarScatter(nn.Module):
        def __init__(self, model_cfg, grid_size, **kwargs):
            super().__init__()

            self.model_cfg = model_cfg
            self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
            self.nx, self.ny, self.nz = grid_size
            assert self.nz == 1

        def forward(self, batch_dict, **kwargs):
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
            batch_spatial_features = []
            batch_size = coords[:, 0].max().int().item() + 1
            for batch_idx in range(batch_size):
                spatial_feature = torch.zeros(
                    self.num_bev_features,
                    self.nz * self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device,
                )

                batch_mask = coords[:, 0] == batch_idx
                this_coords = coords[batch_mask, :]
                indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
                # print('indices 1 dims is max: {}'.format(this_coords[:, 1].max()))
                # print('indices 1 dims is min: {}'.format(this_coords[:, 1].min()))
                indices = indices.type(torch.long)
                pillars = pillar_features[batch_mask, :]
                pillars = pillars.t()
                spatial_feature[:, indices] = pillars
                batch_spatial_features.append(spatial_feature)

            batch_spatial_features = torch.stack(batch_spatial_features, 0)
            batch_spatial_features = batch_spatial_features.view(
                batch_size, self.num_bev_features * self.nz, self.ny, self.nx
            )
            batch_dict['spatial_features'] = batch_spatial_features
            
            ## save features for debug
            # import numpy as np
            # np.save("./jgxue/work/object_detection3d/OpenPCDet/tools/spatial_features.npy", batch_spatial_features.cpu().numpy())
            return batch_dict


    class PointPillarScatter3d(nn.Module):
        def __init__(self, model_cfg, grid_size, **kwargs):
            super().__init__()

            self.model_cfg = model_cfg
            self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
            self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
            self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

        def forward(self, batch_dict, **kwargs):
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

            batch_spatial_features = []
            batch_size = coords[:, 0].max().int().item() + 1
            for batch_idx in range(batch_size):
                spatial_feature = torch.zeros(
                    self.num_bev_features_before_compression,
                    self.nz * self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device,
                )

                batch_mask = coords[:, 0] == batch_idx
                this_coords = coords[batch_mask, :]
                indices = (
                    this_coords[:, 1] * self.ny * self.nx
                    + this_coords[:, 2] * self.nx
                    + this_coords[:, 3]
                )
                indices = indices.type(torch.long)
                pillars = pillar_features[batch_mask, :]
                pillars = pillars.t()
                spatial_feature[:, indices] = pillars
                batch_spatial_features.append(spatial_feature)

            batch_spatial_features = torch.stack(batch_spatial_features, 0)
            batch_spatial_features = batch_spatial_features.view(
                batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx
            )
            batch_dict['spatial_features'] = batch_spatial_features
            return batch_dict


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
            print("mask is:", mask)
            coords = coords[:mask, :]
            pillar_features = pillar_features[:mask, ...]
            batch_spatial_features = []
            batch_size = coords[:, 0].max().int().item() + 1
            for batch_idx in range(batch_size):
                spatial_feature = torch.zeros(
                    ctx.features,
                    ctx.size_x * ctx.size_y * ctx.size_z,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device,
                )

                batch_mask = coords[:, 0] == batch_idx
                this_coords = coords[batch_mask, :]
                indices = (
                    this_coords[:, 1] + this_coords[:, 2] * ctx.size_x + this_coords[:, 3]
                )  # x,y转index z + y*W + x
                indices = indices.type(torch.long)
                pillars = pillar_features[batch_mask, :]
                pillars = pillars.t()  # 转置，C,index
                spatial_feature[:, indices] = pillars
                batch_spatial_features.append(spatial_feature)

            batch_spatial_features = torch.stack(batch_spatial_features, 0)
            batch_spatial_features = batch_spatial_features.view(
                batch_size, ctx.features * ctx.size_z, ctx.size_y, ctx.size_x
            )  # resize得到n,c,h,w的特征图
            return batch_spatial_features

        @staticmethod
        def symbolic(g, *inputs):
            return g.op(
                'vacc::PointPillarScatterFunction',
                inputs[0],
                inputs[1],
                inputs[2],
                sizex_i=inputs[3],
                sizey_i=inputs[4],
                sizez_i=inputs[5],
                features_i=inputs[6],
            )


    ppscatter_function = PointPillarScatterFunction.apply


    # NOTE(lance)
    class PointPillarScatter2(nn.Module):
        def __init__(self, model_cfg, grid_size, **kwargs):
            super().__init__()

            self.model_cfg = model_cfg
            self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
            self.nx, self.ny, self.nz = grid_size
            assert self.nz == 1

        def forward(self, batch_dict, **kwargs):
            pillar_features, coords, mask = (
                batch_dict['pillar_features'],
                batch_dict['voxel_coords'],
                batch_dict['mask'],
            )
            batch_spatial_features = ppscatter_function(
                pillar_features, coords, mask, self.nx, self.ny, self.nz, self.num_bev_features
            )
            # [1,64,496,432]
            batch_dict['spatial_features'] = batch_spatial_features
            return batch_dict
    ```

### modify.3 修改BaseBEVBackbone类实现
- 修改[BaseBEVBackbone](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_2d/base_bev_backbone.py#L81)
    - 建议直接使用此已修改文件：[base_bev_backbone.py](../source_code/onnx_utils/base_bev_backbone.py)，去替换原始同名文件

    ```python
        # export onnx
    def forward(self, spatial_features):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        # spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        # data_dict['spatial_features_2d'] = x
        spatial_features_2d = x

        return spatial_features_2d
    ```

### modify.4 修改AnchorHeadSingle类实现
- 修改[AnchorHeadSingle](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/dense_heads/anchor_head_single.py#L75)，新增如下内容
    - 建议直接使用此已修改文件：[anchor_head_single.py](../source_code/onnx_utils/anchor_head_single.py)，去替换原始同名文件

    ```python
    def forward2(self, data_dict):
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

### modify.5 修改CenterHead类实现
- 修改[CenterHead](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/dense_heads/center_head.py#L385)，替换如下内容
    - 建议直接使用此已修改文件：[center_head.py](../source_code/onnx_utils/center_head.py)，去替换原始同名文件

    ```python
    # export onnx
    def forward(self, spatial_features_2d):
        # spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))
            
        return pred_dicts[0]['center'], pred_dicts[0]['center_z'], pred_dicts[0]['dim'], pred_dicts[0]['rot'], pred_dicts[0]['hm']
    ```

### modify.6 修改PillarVFE类实现
- 修改[CenterPoint](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/detectors/centerpoint.py#L9)，替换如下内容
    - 建议直接使用此已修改文件：[centerpoint.py](../source_code/onnx_utils/centerpoint.py)，去替换原始同名文件

    ```python
    # export onnx
    def forward(self, spatial_features):
        if self.model_cfg['EXPORT_FLAG'] == 0:
            # export PillarVFE  
            cur_module = self.module_list[0]
            out = cur_module(spatial_features)
            return out
        elif self.model_cfg['EXPORT_FLAG'] == 1:
            # export BaseBEVBackbone_CenterHead  
            cur_module = self.module_list[2]
            spatial_features_2d = cur_module(spatial_features)
            cur_module = self.module_list[3]
            out = cur_module(spatial_features_2d)
            return out
        else:
            print("export error")
    ```

### modify.7 修改PointPillar类实现
- 修改[PointPillar](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/detectors/pointpillar.py#L34)，新增如下内容
    - 建议直接使用此已修改文件：[pointpillar.py](../source_code/onnx_utils/pointpillar.py)，去替换原始同名文件

    ```python
    # NOTE(lance)
    def forward2(self, batch_dict):
        # output
        for cur_module in self.module_list[0:4]:
            batch_dict = cur_module(batch_dict)
        return batch_dict['cls_preds'], batch_dict['box_preds'], batch_dict['dir_cls_preds']
    ```


### 以上修改，可通过git-pathch一次更新
- 参见：[center_point.patch](./center_point.patch)

    ```bash
    git clone https://github.com/open-mmlab/OpenPCDet.git
    cd OpenPCDet
    git checkout 8cacccec11db6f59bf6934600c9a175dae254806
    git apply center_point.patch
    ```

### onnx算子注册
- `PointPillarScatterFunction`算子注册，参考：[export.py](../source_code/onnx_utils/export.py)

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
    register_custom_op_symbolic("vacc::PointPillarScatterFunction", symbolic,11)
   ```

### onnx模型导出

- 由于模型的`PointPillarScatter`层是由VDSP实现的，所以模型需要切分成`PillarVFE`部分和`BaseBEVBackbone_CenterHead`部分，分别导出成onnx模型。然后再把模型的两部分和`PointPillarScatterFunction`部分组合到一起，形成一个完整的模型。

- 模型拆分：[export_onnx.py](../source_code/export_onnx.py)
    ```bash
    # for openpcdet_official_deploy.md
    python ../source_code/export_onnx.py \
        --max_voxel_num 32000 \
        --backbone_input_shape 1,64,496,432 \
        --cfg_file ../source_code/config/kitti_centerpoint_pillar_1x_model.yaml \
        --ckpt /path/to/kitti_centerpoint_pillar_1x.pth \
        --save_path ./onnx_models

    # for openpcdet_customer_deploy.md
    python ../source_code/export_onnx.py \
        --max_voxel_num 32000 \
        --backbone_input_shape 1,64,480,480 \
        --cfg_file ../source_code/config/model_centerpoint_pp_stride222_model.yaml \
        --ckpt /path/to/centerpoint-epoch_120.pth \
        --save_path ./onnx_models
    ```
    > 执行成功后将会在save_path目录下生成四个onnx模型文件
    > - PillarVFE.onnx
    > - PillarVFE_sim.onnx
    > - BaseBEVBackbone_CenterHead.onnx
    > - BaseBEVBackbone_CenterHead_sim.onnx

- 模型合并：[merge_onnx.py](../source_code/merge_onnx.py)
    ```bash
    # for openpcdet_official_deploy.md
    python ../source_code/merge_onnx.py \
        --pfe_model_path ./onnx_models/PillarVFE_sim.onnx \
        --rpn_model_path ./onnx_models/BaseBEVBackbone_CenterHead_sim.onnx \
        --max_voxel_num 32000 \
        --backbone_input_shape 1,64,496,432 \
        --target_model_path ./onnx_models/ \
        --target_model_name kitti_centerpoint_pillar_1x.onnx
    
    # for openpcdet_customer_deploy.md
    python ../source_code/merge_onnx.py \
        --pfe_model_path /path/to/PillarVFE_sim.onnx \
        --rpn_model_path /path/to/BaseBEVBackbone_CenterHead_sim.onnx \
        --max_voxel_num 32000 \
        --backbone_input_shape 1,64,480,480 \
        --target_model_path ./onnx_models/ \
        --target_model_name centerpoint.onnx
    ```

    > 命令执行成功后将会在target_model_path目录下生成对应的合并后的onnx模型。
