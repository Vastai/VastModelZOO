# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import vaststreamx as vsx
import numpy as np
attr = vsx.AttrKey
from typing import Union, List
from transformers import AutoTokenizer
from yolo_world_post_process import get_postprocess_gen_grids, filter_scores_and_topk, get_bbox_post_process
def imagetype_to_vsxformat(imagetype):
    if imagetype == 0:
        return vsx.ImageFormat.YUV_NV12
    elif imagetype == 5000:
        return vsx.ImageFormat.RGB_PLANAR
    elif imagetype == 5001:
        return vsx.ImageFormat.BGR_PLANAR
    elif imagetype == 5002:
        return vsx.ImageFormat.RGB_INTERLEAVE
    elif imagetype == 5003:
        return vsx.ImageFormat.BGR_INTERLEAVE
    elif imagetype == 5004:
        return vsx.ImageFormat.GRAY
    else:
        assert False, f"Unrecognize image type {imagetype}"

def bert_get_activation_fp16_A(activation, rep_dtype=None):  # NCHW
    # pdb.set_trace()
    if activation.ndim == 2:
        M, K = activation.shape
        activation = activation.reshape((1, M, K))
    N, M, K = activation.shape

    m_group, k_group = 16, 16

    pad_M, pad_K = M, K
    if M % m_group != 0:
        pad_m = m_group - M % m_group
        pad_M += pad_m

    if K % k_group != 0:
        pad_k = k_group - K % k_group
        pad_K += pad_k

    # tensorize to MK16m16k
    n_num = N
    m_num = pad_M // m_group
    k_num = pad_K // k_group
    block_size = m_group * k_group
    activation = activation.astype(np.float16)
    np_arr = np.zeros((n_num, m_num, k_num, block_size), np.float16)

    for n in range(N):
        for m in range(M):
            for k in range(K):
                addr = (m % m_group) * k_group + (k % k_group)
                np_arr[n, m // m_group, k // k_group, addr] = activation[n, m, k]
    return np_arr

def cv_rgb_image_to_vastai(image_cv, device_id):
    assert len(image_cv.shape) >= 2
    h = image_cv.shape[0]
    w = image_cv.shape[1]
    if len(image_cv.shape) == 3:
        return vsx.create_image(
            image_cv, vsx.ImageFormat.BGR_INTERLEAVE, w, h, device_id
        )
    elif len(image_cv.shape) == 2:
        return vsx.create_image(image_cv, vsx.ImageFormat.GRAY, w, h, device_id)
    else:
        raise Exception("unsupported ndarray shape", image_cv.shape)

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

class YoloWorldImage:
    def __init__(
        self, model_prefix, vdsp_config, batch_size=1, device_id=1, hw_config=""
    ):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0

        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE)
        self.graph_.add_operators(self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

        self.input_shape_ = self.model_.input_shape[0]
        self.preproc_ops_ = vsx.Operator.load_ops_from_json_file(vdsp_config)
        self.fusion_op_ = self.preproc_ops_[0].cast_to_buildin_operator()
        self.oimage_height_, self.oimage_width_ = self.input_shape_[-2:]
        self.fusion_op_.set_attribute(
            {
                attr.OIMAGE_WIDTH: self.oimage_width_,
                attr.OIMAGE_HEIGHT: self.oimage_height_,
            }
        )

    def get_fusion_op_iimage_format(self):
        for op in self.preproc_ops_:
            if op.op_type >= 100:
                buildin_op = op.cast_to_buildin_operator()
                if "IIMAGE_FORMAT" in list(buildin_op.attributes.keys()):
                    imagetype = buildin_op.get_attribute(attr.IIMAGE_FORMAT)
                    return imagetype_to_vsxformat(imagetype)
                else:
                    return vsx.ImageFormat.YUV_NV12
        assert False, "Can't find fusion op that op_type >= 100"

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height = input_shape[-2]
        width = input_shape[-1]
        dummy_image = np.zeros((height, width, 3), dtype=dtype)
        dummy_txt = np.zeros((1203, 512), dtype=np.float32)
        dummy_txt = bert_get_activation_fp16_A(dummy_txt)
        if ctx == "CPU":
            return ([dummy_image] * batch_size, [dummy_txt] * batch_size)
        else:
            vacc_image = vsx.create_image(
                dummy_image,
                vsx.ImageFormat.BGR_INTERLEAVE,
                width,
                height,
                self.device_id_,
            )
            vacc_txt = vsx.from_numpy(dummy_txt, self.device_id_)
            return ([vacc_image] * batch_size, [vacc_txt] * batch_size)

    def process(self, input):
        image, tensor = input
        vacc_images = []
        if isinstance(image, list):
            if isinstance(image[0], np.ndarray):
                vacc_images = [
                    cv_rgb_image_to_vastai(x, self.device_id_) for x in image
                ]
            else:
                vacc_images = image
        else:
            if isinstance(image, np.ndarray):
                vacc_images.append(cv_rgb_image_to_vastai(image, self.device_id_))
            else:
                vacc_images.append(image)

        vacc_tensors = []
        if isinstance(tensor, list):
            if isinstance(tensor[0], np.ndarray):
                vacc_tensors = [vsx.from_numpy(t, self.device_id_) for t in tensor]
            else:
                vacc_tensors = tensor
        else:
            if isinstance(tensor, np.ndarray):
                vacc_tensors.append(vsx.from_numpy(tensor, self.device_id_))
            else:
                vacc_tensors.append(tensor)
        res = self.process_impl(vacc_images, vacc_tensors)
        if isinstance(image, list):
            return res
        else:
            return res[0]

    def process_impl(self, images, tensors):
        preproc_images = []
        for image in images:
            height, width = image.height, image.width
            self.fusion_op_.set_attribute(
                {
                    attr.IIMAGE_WIDTH: width,
                    attr.IIMAGE_WIDTH_PITCH: width,
                    attr.IIMAGE_HEIGHT: height,
                    attr.IIMAGE_HEIGHT_PITCH: height,
                }
            )
            vdsp_out = self.fusion_op_.execute(
                tensors=[image],
                output_info=[(([1, 160, 160, 1, 256]), vsx.TypeFlag.FLOAT16)],
            )[0]

            preproc_images.append(vdsp_out)

        inputs = [
            [vdsp_out, txt_out] for vdsp_out, txt_out in zip(preproc_images, tensors)
        ]
        outputs = self.stream_.run_sync(inputs)

        results = []
        for output in outputs:
            outs = [vsx.as_numpy(out).astype(np.float32) for out in output]
            result = []
            for i, out in enumerate(outs):
                if i < 3:
                    result.append(out)
                else:
                    out = np.reshape(out, newshape=(1, -1, 4))
                    out = np.transpose(out, axes=(0, 2, 1))
                    out = np.reshape(
                        out, newshape=(1, 4, int(np.sqrt(out.shape[2])), -1)
                    )
                    result.append(out)
            results.append(result)

        return results

    @property
    def batch_size(self):
        return self.model_.batch_size

    @property
    def max_batch_size(self):
        return self.model_.max_batch_size

    @property
    def input_count(self):
        return self.model_.input_count

    @property
    def output_count(self):
        return self.model_.output_count

    @property
    def input_shape(self):
        return self.model_.input_shape

    @property
    def output_shape(self):
        return self.model_.output_shape

class YoloWorldText():
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        tokenizer_path,
        batch_size=1,
        device_id=0,
        hw_config="",
    ):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.preproc_ops_ = vsx.Operator.load_ops_from_json_file(vdsp_config)
        assert len(self.preproc_ops_) > 0, "load_ops_from_json_file failed"
        self.graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE)
        self.graph_.add_operators(*self.preproc_ops_, self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()
        self.bytes_size_ = self.model_.input_count * 4
        self.tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_path)

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        tokens = self.make_tokens("test string")
        if ctx == "CPU":
            return [tokens] * batch_size
        else:
            return [
                [vsx.from_numpy(token, self.device_id_) for token in tokens]
            ] * batch_size

    def make_tokens(self, text):
        assert isinstance(text, str), f"input type must be str"
        token_dict = self.tokenizer_(text=text, return_tensors="pt", padding=True)
        token = token_dict["input_ids"][0]
        input_seq_len = 16
        token_padding = np.full([input_seq_len], 49407, dtype=np.int32)  # pad
        token_padding[: len(token)] = token
        # make mask
        token_mask = np.ones(shape=(input_seq_len), dtype=np.int32) * (-1)
        mask = token_dict["attention_mask"][0]
        token_mask[: len(mask)] = mask
        # make input
        zero_arr = np.zeros(token_padding.shape, dtype=np.int32)
        tokens = []
        tokens.append(token_padding)
        tokens.append(zero_arr)
        tokens.append(zero_arr)
        tokens.append(token_mask)
        tokens.append(zero_arr)
        tokens.append(zero_arr)

        return tokens

    def process(
        self,
        input: Union[
            List[List[vsx.Tensor]],
            List[List[np.ndarray]],
            List[vsx.Tensor],
            List[np.ndarray],
            List[str],
            str,
        ],
    ):
        if isinstance(input, list):
            if isinstance(input[0], list):
                if isinstance(input[0][0], np.ndarray):
                    return self.process(
                        [
                            [
                                vsx.from_numpy(
                                    np.array(x, dtype=np.int32), self.device_id_
                                )
                                for x in one
                            ]
                            for one in input
                        ]
                    )
                else:
                    return self.process_impl(input)
            elif isinstance(input[0], str):
                return self.process([self.make_tokens(x) for x in input])
            elif isinstance(input[0], np.ndarray):
                tensors = [
                    vsx.from_numpy(np.array(x, dtype=np.int32), self.device_id_)
                    for x in input
                ]
                return self.process_impl([tensors])[0]
            else:
                return self.process_impl([input])[0]
        else:
            tokens = self.make_tokens(input)
            return self.process(tokens)

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [
            [vsx.as_numpy(out).astype(np.float32) for out in output]
            for output in outputs
        ]

    @property
    def batch_size(self):
        return self.model_.batch_size

    @property
    def max_batch_size(self):
        return self.model_.max_batch_size

    @property
    def input_count(self):
        return self.model_.input_count

    @property
    def output_count(self):
        return self.model_.output_count

    @property
    def input_shape(self):
        return self.model_.input_shape

    @property
    def output_shape(self):
        return self.model_.output_shape


class YoloWorld:
    def __init__(
        self,
        imgmod_prefix,
        imgmod_vdsp_config,
        txtmod_prefix,
        txtmod_vdsp_config,
        tokenizer_path,
        batch_size=1,
        device_id=0,
        score_thres=0.001,
        nms_pre=30000,
        iou_thres=0.7,
        max_per_image=300,
        imgmod_hw_config="",
        txtmod_hw_config="",
    ) -> None:
        self.imgmod_ = YoloWorldImage(
            imgmod_prefix,
            imgmod_vdsp_config,
            batch_size,
            device_id,
            imgmod_hw_config,
        )
        self.txtmod_ = YoloWorldText(
            txtmod_prefix,
            txtmod_vdsp_config,
            tokenizer_path,
            batch_size,
            device_id,
            txtmod_hw_config,
        )
        self.device_id_ = device_id
        self.score_thres_ = score_thres
        self.nms_pre_ = nms_pre
        self.iou_thres_ = iou_thres
        self.max_per_image_ = max_per_image

    def get_fusion_op_iimage_format(self):
        return self.imgmod_.get_fusion_op_iimage_format()

    def process(self, image, texts):
        text_features = self.process_texts(texts)
        return self.process_image(image, text_features)

    def process_texts(self, texts):
        txt_features = self.txtmod_.process(texts)
        text_feature = np.array(txt_features).squeeze()
        return (text_feature, bert_get_activation_fp16_A(text_feature))

    def process_image(self, image, text_features):
        text_feature, text_884_feature = text_features
        img_features = self.imgmod_.process((image, text_884_feature))
        return self.post_process(image, img_features, text_feature)

    def post_process(self, image, img_features, txt_features):
        score_160, score_80, score_40 = get_scores_batch(img_features, txt_features)

        if isinstance(image, np.ndarray):
            ori_shape = (image.shape[0], image.shape[1])
        else:
            ori_shape = (image.height, image.width)

        imgmod_shape = self.imgmod_.input_shape[0]
        new_shape = (imgmod_shape[-2], imgmod_shape[-1])
        r = min(new_shape[0] / ori_shape[0], new_shape[1] / ori_shape[1])

        ratio = r, r  # width, height ratios
        new_unpad = int(round(ori_shape[1] * r)), int(round(ori_shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        pad = (dw, dh)

        res = get_postprocess(
            (score_160, score_80, score_40),
            (img_features[-3:]),
            ori_shape=ori_shape,
            scale_factor=ratio,
            pad_param=(pad[1], pad[1], pad[0], pad[0]),
            score_thr=self.score_thres_,
            nms_pre=self.nms_pre_,
            iou_threshold=self.iou_thres_,
            cfg_max_per_img=self.max_per_image_,
        )

        scores = res["scores"].numpy()
        bboxes = res["bboxes"].numpy()
        labels = res["labels"].numpy()

        result = {"scores": scores, "bboxes": bboxes, "labels": labels}
        return result

    def compute_tokens(self, text):
        assert isinstance(text, str), f"input text must be str"
        return self.txtmod_.make_tokens(text=text)
