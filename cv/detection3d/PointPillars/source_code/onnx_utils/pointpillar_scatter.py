# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

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

# class PointPillarScatter(nn.Module):
#     def __init__(self, model_cfg, grid_size, **kwargs):
#         super().__init__()

#         self.model_cfg = model_cfg
#         self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
#         self.nx, self.ny, self.nz = grid_size
#         assert self.nz == 1

#     def forward(self, batch_dict, **kwargs):
#         pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
#         batch_spatial_features = []
#         batch_size = coords[:, 0].max().int().item() + 1
#         for batch_idx in range(batch_size):
#             spatial_feature = torch.zeros(
#                 self.num_bev_features,
#                 self.nz * self.nx * self.ny,
#                 dtype=pillar_features.dtype,
#                 device=pillar_features.device)

#             batch_mask = coords[:, 0] == batch_idx
#             this_coords = coords[batch_mask, :]
#             indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]       # x,y转index z + y*W + x
#             indices = indices.type(torch.long)
#             pillars = pillar_features[batch_mask, :]
#             pillars = pillars.t()   #转置，C,index
#             spatial_feature[:, indices] = pillars
#             batch_spatial_features.append(spatial_feature)

#         batch_spatial_features = torch.stack(batch_spatial_features, 0)
#         batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx) # resize得到n,c,h,w的特征图
#         #[1,64,496,432]
#         batch_dict['spatial_features'] = batch_spatial_features
#         return batch_dict
