
import torch
import numpy as np

from mmcv.ops.nms import nms
from concurrent.futures import ThreadPoolExecutor
import time


def get_bbox_post_process(results, cfg_nms, cfg_max_per_img, with_nms=True):
    def batched_nms(
        boxes, scores, idxs, cfg_nms, cfg_max_per_img, class_agnostic=False
    ):
        nms_cfg_ = cfg_nms.copy()
        class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
        if class_agnostic:
            boxes_for_nms = boxes
        else:
            if boxes.size(-1) == 5:
                max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
                offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
                boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
                boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]], dim=-1)
            else:
                max_coordinate = boxes.max()
                offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
                boxes_for_nms = boxes + offsets[:, None]

        nms_type = nms_cfg_.pop("type", "nms")
        nms_op = eval(nms_type)
        # nms_op = nms

        split_thr = nms_cfg_.pop("split_thr", 10000)
        if boxes_for_nms.shape[0] < split_thr:
            dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
            boxes = boxes[keep]
            scores = dets[:, -1]
        else:
            max_num = nms_cfg_.pop("max_num", -1)
            total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
            # Some type of nms would reweight the score, such as SoftNMS
            scores_after_nms = scores.new_zeros(scores.size())
            for id in torch.unique(idxs):
                mask = (idxs == id).nonzero(as_tuple=False).view(-1)
                dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
                total_mask[mask[keep]] = True
                scores_after_nms[mask[keep]] = dets[:, -1]
            keep = total_mask.nonzero(as_tuple=False).view(-1)

            scores, inds = scores_after_nms[keep].sort(descending=True)
            keep = keep[inds]
            boxes = boxes[keep]

            if max_num > 0:
                keep = keep[:max_num]
                boxes = boxes[:max_num]
                scores = scores[:max_num]

        boxes = torch.cat([boxes, scores[:, None]], -1)
        return boxes, keep

    if with_nms and results["bboxes"].shape[0] > 0:
        bboxes = results["bboxes"]
        det_bboxes, keep_idxs = batched_nms(
            bboxes, results["scores"], results["labels"], cfg_nms, cfg_max_per_img
        )
        results["scores"] = results["scores"][keep_idxs]
        results["bboxes"] = results["bboxes"][keep_idxs]
        results["labels"] = results["labels"][keep_idxs]
        # some nms would reweight the score, such as softnms
        results["scores"] = det_bboxes[:, -1]
        results["scores"] = results["scores"][:cfg_max_per_img]
        results["bboxes"] = results["bboxes"][:cfg_max_per_img, :]
        results["labels"] = results["labels"][:cfg_max_per_img]
    return results


def get_postprocess_gen_grids(
    cls_scores, bbox_preds, num_classes=1203, featmap_strides=[8, 16, 32]
):
    def get_bbox_coder_decode(points, pred_bboxes, stride, max_shape=None):
        def distance2bbox(points, distance, max_shape=None):
            x1 = points[..., 0] - distance[..., 0]
            y1 = points[..., 1] - distance[..., 1]
            x2 = points[..., 0] + distance[..., 2]
            y2 = points[..., 1] + distance[..., 3]
            bboxes = torch.stack([x1, y1, x2, y2], -1)
            return bboxes

        def decode(points, pred_bboxes, stride, max_shape=None):
            assert points.size(-2) == pred_bboxes.size(-2)
            assert points.size(-1) == 2
            assert pred_bboxes.size(-1) == 4
            clip_border = True
            if clip_border is False:
                max_shape = None
            pred_bboxes = pred_bboxes * stride[None, :, None]
            return distance2bbox(points, pred_bboxes, max_shape)

        return decode(points, pred_bboxes, stride, max_shape=max_shape)

    def get_grid_priors(featmap_sizes, strides):
        strides = [tuple(np.array(e).repeat(2)) for e in strides]
        offset = 0.5

        def _meshgrid(x, y, row_major=True):
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            if row_major:
                # warning .flatten() would cause error in ONNX exporting
                # have to use reshape here
                return xx.reshape(-1), yy.reshape(-1)

            else:
                return yy.reshape(-1), xx.reshape(-1)

        def single_level_grid_priors(
            featmap_size, level_idx, dtype, device, with_stride
        ):
            # strides = [8, 16, 32]
            # offset = 0.5
            feat_h, feat_w = featmap_size
            stride_w, stride_h = strides[level_idx]
            shift_x = (torch.arange(0, feat_w, device=device) + offset) * stride_w
            # keep featmap_size as Tensor instead of int, so that we
            # can convert to ONNX correctly
            shift_x = shift_x.to(torch.float32)

            shift_y = (torch.arange(0, feat_h, device=device) + offset) * stride_h
            # keep featmap_size as Tensor instead of int, so that we
            # can convert to ONNX correctly
            shift_y = shift_y.to(torch.float32)
            shift_xx, shift_yy = _meshgrid(shift_x, shift_y)

            if not with_stride:
                shifts = torch.stack([shift_xx, shift_yy], dim=-1)
            else:
                # use `shape[0]` instead of `len(shift_xx)` for ONNX export
                stride_w = shift_xx.new_full((shift_xx.shape[0],), stride_w).to(dtype)
                stride_h = shift_xx.new_full((shift_yy.shape[0],), stride_h).to(dtype)
                shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h], dim=-1)
            all_points = shifts.to(device)
            return all_points

        def grid_priors(
            featmap_sizes, dtype="float32", device="cpu", with_stride=False
        ):
            multi_level_priors = []
            num_levels = len(featmap_sizes)
            for i in range(num_levels):
                priors = single_level_grid_priors(
                    featmap_sizes[i],
                    level_idx=i,
                    dtype=dtype,
                    device=device,
                    with_stride=with_stride,
                )
                multi_level_priors.append(priors)
            return multi_level_priors

        return grid_priors(featmap_sizes)

    # num_imgs = 1
    num_base_priors = 1

    featmap_sizes = [
        cls_score.shape[2:] for cls_score in cls_scores
    ]  # [(160, 160), (80, 80), (40, 40)]
    mlvl_priors = get_grid_priors(featmap_sizes, featmap_strides)
    flatten_priors = torch.cat(mlvl_priors)

    mlvl_strides = [
        flatten_priors.new_full(
            (featmap_size[0] * featmap_size[1] * num_base_priors,), stride
        )
        for featmap_size, stride in zip(featmap_sizes, featmap_strides)
    ]  # self.featmap_strides [8, 16, 32],
    flatten_stride = torch.cat(mlvl_strides)

    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(1, -1, num_classes)
        for cls_score in cls_scores
    ]
    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(1, -1, 4) for bbox_pred in bbox_preds
    ]

    flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid().squeeze()
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

    flatten_decoded_bboxes = get_bbox_coder_decode(
        flatten_priors[None], flatten_bbox_preds, flatten_stride
    )

    flatten_objectness = [None for _ in range(1)]
    return flatten_cls_scores, flatten_decoded_bboxes.squeeze(), flatten_objectness


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)
    num_topk = min(topk, valid_idxs.size(0))

    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    return scores, labels, keep_idxs


def get_postprocess(
    cls_scores,
    bbox_preds,
    ori_shape=(427, 640),
    scale_factor=(2.0, 2.0),
    pad_param=(213, 213, 0, 0),
    score_thr=0.001,
    nms_pre=30000,
    iou_threshold=0.7,
    cfg_max_per_img=300,
):
    cls_scores = [torch.from_numpy(cls_score) for cls_score in cls_scores]
    bbox_preds = [torch.from_numpy(bbox_pred) for bbox_pred in bbox_preds]

    flatten_cls_scores, flatten_decoded_bboxes, flatten_objectness = (
        get_postprocess_gen_grids(cls_scores, bbox_preds)
    )

    scores, labels, keep_idxs = filter_scores_and_topk(
        flatten_cls_scores, score_thr, nms_pre
    )

    bboxes = flatten_decoded_bboxes[keep_idxs]
    bboxes -= torch.from_numpy(
        np.array([pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
    )
    bboxes /= torch.from_numpy(np.array(scale_factor)).repeat((1, 2))

    results = {"scores": scores, "bboxes": bboxes, "labels": labels}

    cfg_nms = {"type": "nms", "iou_threshold": iou_threshold}
    cfg_max_per_img = cfg_max_per_img
    results = get_bbox_post_process(results, cfg_nms, cfg_max_per_img)

    results["bboxes"][:, 0::2].clamp_(0, ori_shape[1])
    results["bboxes"][:, 1::2].clamp_(0, ori_shape[0])

    return results


def get_scores_batch(img_features, text_feature):
    data_3489, data_3566, data_3643 = img_features[:3]

    def get_l2norm(data, axis=1):
        x0 = np.abs(data)
        return np.divide(data, np.sqrt(np.sum(x0 * x0, axis=axis, keepdims=True)))

    def get_res(data_a, data_b, h, w, multiply_const, add_const):
        data_a = np.transpose(data_a, axes=(0, 2, 3, 1)).reshape((-1, 512))
        res = np.matmul(data_a, data_b)
        res = np.reshape(res, (1, h, w, -1)).transpose((0, 3, 1, 2))
        res = np.multiply(res, multiply_const)
        res = np.add(res, add_const)
        return res

    res_l2_b = get_l2norm(text_feature).transpose()

    with ThreadPoolExecutor() as executor:
        res_3489 = executor.submit(
            get_res,
            data_3489,
            res_l2_b,
            160,
            160,
            1.7583335638046265,
            -12.202274322509766,
        )
        res_3566 = executor.submit(
            get_res,
            data_3566,
            res_l2_b,
            80,
            80,
            1.7852298021316528,
            -10.711109161376953,
        )
        res_3643 = executor.submit(
            get_res,
            data_3643,
            res_l2_b,
            40,
            40,
            2.0692732334136963,
            -9.184325218200684,
        )

    return res_3489.result(), res_3566.result(), res_3643.result()
