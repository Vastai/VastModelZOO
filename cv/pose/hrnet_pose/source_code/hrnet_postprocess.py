import torch
# import numpy as np

from utils.hrnet import HRNet
from utils.ae_simple_head import AESimpleHead
from utils.group import HeatmapParser
from utils.bottom_up_eval import (get_group_preds, aggregate_stage_flip, aggregate_scale, flip_feature_maps,
                                split_ae_outputs)
# from mmpose.core.evaluation import (aggregate_stage_flip, get_group_preds, flip_feature_maps,
#                                     split_ae_outputs, aggregate_scale)

channel_cfg = dict(
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])
test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=30,
        scale_factor=[1],
        with_heatmaps=[True],
        with_ae=[True],
        project2image=True,
        align_corners=False,
        nms_kernel=5,
        nms_padding=2,
        tag_per_joint=True,
        detection_threshold=0.1,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=True,
        flip_test=False)

backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))))

keypoint_head=dict(
        type='AESimpleHead',
        in_channels=32,
        num_joints=17,
        num_deconv_layers=0,
        tag_per_joint=True,
        with_ae_loss=[True],
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0]))

new_backbone = HRNet(backbone['extra'],in_channels=3)
new_keypoint_head = AESimpleHead(in_channels=32,
        num_joints=17,
        num_deconv_layers=0,
        tag_per_joint=True,
        with_ae_loss=[True],
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0])
                            )

def forward_test(outputs, outputs_flipped, img_metas, return_heatmap=False, flip_test= True, **kwargs):
    """Inference the bottom-up model.

    Note:
        - Batchsize: N (currently support batchsize = 1)
        - num_img_channel: C
        - img_width: imgW
        - img_height: imgH

    Args:
        flip_index (List(int)):
        aug_data (List(Tensor[NxCximgHximgW])): Multi-scale image
        test_scale_factor (List(float)): Multi-scale factor
        base_size (Tuple(int)): Base size of image when scale is 1
        center (np.ndarray): center of image
        scale (np.ndarray): the scale of image
    """
    # assert img.size(0) == 1
    # assert len(img_metas) == 1

    img_metas = img_metas

    # aug_data = img_metas['aug_data']

    test_scale_factor = img_metas['test_scale_factor']
    base_size = img_metas['base_size']
    center = img_metas['center']
    scale = img_metas['scale']

    result = {}

    scale_heatmaps_list = []
    scale_tags_list = []

    for idx, s in enumerate(sorted(test_scale_factor, reverse=True)):
        # image_resized = aug_data[idx].to(img.device)

        # features = new_backbone(image_resized)
        
        # outputs = new_keypoint_head(features)
        heatmaps, tags = split_ae_outputs(
            outputs, test_cfg['num_joints'],
            test_cfg['with_heatmaps'], test_cfg['with_ae'],
            test_cfg.get('select_output_index', range(len(outputs))))
        if  flip_test:
            # use flip test
            # features_flipped = new_backbone(
            #     torch.flip(image_resized, [3]))
            # outputs_flipped = new_keypoint_head(features_flipped)

            heatmaps_flipped, tags_flipped = split_ae_outputs(
                outputs_flipped, test_cfg['num_joints'],
                test_cfg['with_heatmaps'], test_cfg['with_ae'],
                test_cfg.get('select_output_index',
                                    range(len(outputs))))

            heatmaps_flipped = flip_feature_maps(
                heatmaps_flipped, flip_index=img_metas['flip_index'])
            if test_cfg['tag_per_joint']:
                tags_flipped = flip_feature_maps(
                    tags_flipped, flip_index=img_metas['flip_index'])
            else:
                tags_flipped = flip_feature_maps(
                    tags_flipped, flip_index=None, flip_output=True)

        else:
            heatmaps_flipped = None
            tags_flipped = None
        aggregated_heatmaps = aggregate_stage_flip(
            heatmaps,
            heatmaps_flipped,
            index=-1,
            project2image=test_cfg['project2image'],
            size_projected=base_size,
            align_corners=test_cfg.get('align_corners', True),
            aggregate_stage='average',
            aggregate_flip='average')

        aggregated_tags = aggregate_stage_flip(
            tags,
            tags_flipped,
            index=-1,
            project2image=test_cfg['project2image'],
            size_projected=base_size,
            align_corners=test_cfg.get('align_corners', True),
            aggregate_stage='concat',
            aggregate_flip='concat')

        if s == 1 or len(test_scale_factor) == 1:
            if isinstance(aggregated_tags, list):
                scale_tags_list.extend(aggregated_tags)
            else:
                scale_tags_list.append(aggregated_tags)
        if isinstance(aggregated_heatmaps, list):
            scale_heatmaps_list.extend(aggregated_heatmaps)
        else:
            scale_heatmaps_list.append(aggregated_heatmaps)
        
    aggregated_heatmaps = aggregate_scale(
        scale_heatmaps_list,
        align_corners=test_cfg.get('align_corners', True),
        aggregate_scale='average')
    
    aggregated_tags = aggregate_scale(
        scale_tags_list,
        align_corners=test_cfg.get('align_corners', True),
        aggregate_scale='unsqueeze_concat')
    

    heatmap_size = aggregated_heatmaps.shape[2:4]
    tag_size = aggregated_tags.shape[2:4]
    if heatmap_size != tag_size:
        tmp = []
        for idx in range(aggregated_tags.shape[-1]):
            tmp.append(
                torch.nn.functional.interpolate(
                    aggregated_tags[..., idx],
                    size=heatmap_size,
                    mode='bilinear',
                    align_corners=test_cfg.get('align_corners',
                                                    True)).unsqueeze(-1))
        aggregated_tags = torch.cat(tmp, dim=-1)

    # perform grouping
    new_parser = HeatmapParser(test_cfg)
    grouped, scores = new_parser.parse(aggregated_heatmaps,
                                        aggregated_tags,
                                        # self.test_cfg['adjust'],
                                        # self.test_cfg['refine'])
                                        True,
                                        True)
    preds = get_group_preds(
        grouped,
        center,
        scale, [aggregated_heatmaps.size(3),
                aggregated_heatmaps.size(2)],
        # use_udp=self.use_udp)
        use_udp=False)
    image_paths = []
    image_paths.append(img_metas['image_file'])

    if return_heatmap:
        output_heatmap = aggregated_heatmaps.detach().cpu().numpy()
    else:
        output_heatmap = None

    result['preds'] = preds
    result['scores'] = scores
    result['image_paths'] = image_paths
    result['output_heatmap'] = output_heatmap

    return result