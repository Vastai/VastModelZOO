
import vaststreamx as vsx
import numpy as np
import torch
from torchvision.ops import box_convert
from typing import Dict, List, Union
from transformers import AutoTokenizer
attr = vsx.AttrKey

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

def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list):
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token), device=input_ids.device).bool()
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token, device=input_ids.device)
        .bool()
        .unsqueeze(0)
        .repeat(bs, 1, 1)
    )
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    cate_to_token_mask_list = [[] for _ in range(bs)]
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[
                row, previous_col + 1 : col + 1, previous_col + 1 : col + 1
            ] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )
            c2t_maski = torch.zeros((num_token), device=input_ids.device).bool()
            c2t_maski[previous_col + 1 : col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col

    cate_to_token_mask_list = [
        torch.stack(cate_to_token_mask_listi, dim=0)
        for cate_to_token_mask_listi in cate_to_token_mask_list
    ]
    return attention_mask, position_ids.to(torch.long)

class ModelBase:
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        output_type=vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE,
    ) -> None:
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.preproc_ops_ = vsx.Operator.load_ops_from_json_file(vdsp_config)
        assert len(self.preproc_ops_) > 0, "load_ops_from_json_file failed"
        self.graph_ = vsx.Graph(output_type)
        self.graph_.add_operators(*self.preproc_ops_, self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

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

class ModelCV(ModelBase):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        output_type=vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE,
    ) -> None:
        super().__init__(
            model_prefix, vdsp_config, batch_size, device_id, hw_config, output_type
        )

    def process(
        self, input: Union[List[np.ndarray], List[vsx.Image], np.ndarray, vsx.Image]
    ):
        if isinstance(input, list):
            if isinstance(input[0], np.ndarray):
                return self.process(
                    [cv_rgb_image_to_vastai(x, self.device_id_) for x in input]
                )
            else:
                return self.process_impl(input)
        return self.process([input])[0]

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height = input_shape[-2]
        width = input_shape[-1]
        dummy = np.zeros((height, width, 3), dtype=dtype)
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            vacc_dummy = vsx.create_image(
                dummy, vsx.ImageFormat.BGR_INTERLEAVE, width, height, self.device_id_
            )
            return [vacc_dummy] * batch_size

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [[vsx.as_numpy(o) for o in out] for out in outputs]
        # return [vsx.as_numpy(out[0]) for out in outputs]

class ModelNLP(ModelBase):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        output_type=vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE,
    ) -> None:
        super().__init__(
            model_prefix,
            vdsp_config,
            batch_size,
            device_id,
            hw_config,
            output_type,
        )
        self.bytes_size_ = self.model_.input_count * 4

    def process(
        self,
        input: Union[
            List[List[np.ndarray]],
            List[List[vsx.Tensor]],
            List[np.ndarray],
            List[vsx.Tensor],
        ],
    ):
        if isinstance(input[0], list):
            if isinstance(input[0][0], np.ndarray):
                return self.process(
                    [
                        [
                            vsx.from_numpy(np.array(x), self.device_id_)
                            for x in one_input
                        ]
                        for one_input in input
                    ]
                )
            else:
                return self.process_impl(input)
        return self.process([input])[0]

    def process_impl(self, input):
        raise NotImplementedError(
            "pure virtual function must be overridden in derived classes"
        )

class GroundingDinoText(ModelNLP):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        tokenizer_path,
        batch_size=1,
        device_id=0,
        hw_config="",
    ):
        super().__init__(
            model_prefix=model_prefix,
            vdsp_config=vdsp_config,
            batch_size=batch_size,
            device_id=device_id,
            hw_config=hw_config,
        )
        self.tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_path)
        self.specical_tokens = self.tokenizer_.convert_tokens_to_ids(
            ["[CLS]", "[SEP]", ".", "?"]
        )

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        if ctx == "CPU":
            return [
                [np.zeros(shape, dtype=dtype) for shape in input_shape]
            ] * batch_size
        else:
            return [
                [
                    vsx.from_numpy(np.zeros(shape, dtype=dtype), self.device_id_)
                    for shape in input_shape
                ]
            ] * batch_size

    def text_tokenization(self, text):
        assert isinstance(text, str), f"input type must be str"
        token_dict = self.tokenizer_(text=text, return_tensors="pt", padding="longest")
        token = token_dict["input_ids"][0]
        input_len = 208  # model_input_len
        seq_len = len(token)
        assert (
            seq_len <= input_len
        ), f"token len=({seq_len}) is larger than model max len=({input_len}),please input shorter string"

        # input_ids
        input_ids = np.full([1, input_len], 0, dtype=np.int32)  # pad
        input_ids[0, : len(token)] = token

        # make mask
        token_mask, text_position_ids = (
            generate_masks_with_special_tokens_and_transfer_map(
                token_dict, self.specical_tokens
            )
        )
        token_mask = token_mask.numpy()
        text_position_ids = text_position_ids.numpy()

        # position_ids
        position_ids = np.full([1, input_len], 0, dtype=np.int32)  # pad
        position_ids[:, :seq_len] = text_position_ids

        # token_type_ids
        token_type_ids = np.zeros((1, input_len), dtype=np.int32)
        token_type_ids[0, :seq_len] = token_dict["token_type_ids"][0]

        # attention_mask
        text_attention_mask = token_dict["attention_mask"].numpy()
        attention_mask = np.zeros(input_ids.shape, dtype=np.int32)
        attention_mask[0, :195] = text_attention_mask[0, :195]

        # text_token_mask
        text_token_mask = np.zeros((208, 208), dtype=np.int32)
        text_token_mask[:195, :195] = token_mask.astype(np.int32)
        text_token_mask[text_token_mask == 0] = -10000
        text_token_mask[text_token_mask == 1] = 0
        text_token_mask = text_token_mask.astype(np.float16)

        # make input
        tokens = []
        tokens.append(input_ids)
        tokens.append(position_ids)
        tokens.append(token_type_ids)
        tokens.append([])  # attention mask
        tokens.append(text_token_mask)  # memory
        tokens.append([])  # cache

        return [tokens, attention_mask]

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [[vsx.as_numpy(out) for out in output] for output in outputs]

class GroundingDinoImage(ModelCV):
    def __init__(
        self, model_prefix, vdsp_config, batch_size=1, device_id=0, hw_config=""
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [[vsx.as_numpy(out) for out in output] for output in outputs]


class GroundingDinoDecoder:
    def __init__(self, model_prefix, batch_size=1, device_id=0, hw_config=""):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE)
        self.graph_.add_operators(self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

    def process(self, img_encoded, txt_encoded, tokens, attention_mask):
        img_encoded = np.array(img_encoded).squeeze()
        txt_encoded = np.array(txt_encoded).squeeze()

        img_encoded = img_encoded[:22223, :]

        position_ids, text_token_mask = tokens[1], tokens[4]

        text_token_mask[text_token_mask == 0] = 1
        text_token_mask[text_token_mask == -10000] = 0

        attention_mask = attention_mask.astype(np.float16)
        attention_mask[attention_mask == 0.0] = float("-inf")
        attention_mask[attention_mask == 1.0] = 0.0

        inputs = []
        inputs.append(vsx.from_numpy(img_encoded, self.device_id_))
        inputs.append(vsx.from_numpy(txt_encoded, self.device_id_))
        inputs.append(vsx.from_numpy(attention_mask, self.device_id_))
        inputs.append(vsx.from_numpy(position_ids, self.device_id_))
        inputs.append(vsx.from_numpy(text_token_mask, self.device_id_))

        outputs = self.stream_.run_sync([inputs])[0]
        return [vsx.as_numpy(out).astype(np.float32) for out in outputs]


class GroundingDino:
    def __init__(
        self,
        txtmod_prefix,
        txtmod_vdsp_params,
        imgmod_prefix,
        imgmod_vdsp_params,
        decmod_prefix,
        tokenizer_path,
        label_path,
        batch_size=1,
        device_id=0,
        threshold=0.25,
        txtmod_hw_config="",
        imgmod_hw_config="",
        decmod_hw_config="",
    ):
        self.text_model_ = GroundingDinoText(
            txtmod_prefix,
            txtmod_vdsp_params,
            tokenizer_path,
            batch_size,
            device_id,
            txtmod_hw_config,
        )
        self.image_model_ = GroundingDinoImage(
            imgmod_prefix, imgmod_vdsp_params, batch_size, device_id, imgmod_hw_config
        )
        self.decoder_model_ = GroundingDinoDecoder(
            decmod_prefix, batch_size, device_id, decmod_hw_config
        )
        self.device_id_ = device_id
        self.threshold_ = threshold

        id_map = {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 7,
            7: 8,
            8: 9,
            9: 10,
            10: 11,
            11: 13,
            12: 14,
            13: 15,
            14: 16,
            15: 17,
            16: 18,
            17: 19,
            18: 20,
            19: 21,
            20: 22,
            21: 23,
            22: 24,
            23: 25,
            24: 27,
            25: 28,
            26: 31,
            27: 32,
            28: 33,
            29: 34,
            30: 35,
            31: 36,
            32: 37,
            33: 38,
            34: 39,
            35: 40,
            36: 41,
            37: 42,
            38: 43,
            39: 44,
            40: 46,
            41: 47,
            42: 48,
            43: 49,
            44: 50,
            45: 51,
            46: 52,
            47: 53,
            48: 54,
            49: 55,
            50: 56,
            51: 57,
            52: 58,
            53: 59,
            54: 60,
            55: 61,
            56: 62,
            57: 63,
            58: 64,
            59: 65,
            60: 67,
            61: 70,
            62: 72,
            63: 73,
            64: 74,
            65: 75,
            66: 76,
            67: 77,
            68: 78,
            69: 79,
            70: 80,
            71: 81,
            72: 82,
            73: 84,
            74: 85,
            75: 86,
            76: 87,
            77: 88,
            78: 89,
            79: 90,
        }
        with open(label_path) as fin:
            self.category_dict = {
                id_map[i]: line.strip() for i, line in enumerate(fin.readlines())
            }
        cat_list = [val for val in self.category_dict.values()]
        captions, cat2tokenspan = self.build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = self.create_positive_map_from_span(
            self.text_model_.tokenizer_(captions), tokenspanlist
        )  # 80, 256. normed
        # build a mapping from label_id to pos_map
        new_pos_map = torch.zeros((91, 256))
        for k, v in id_map.items():
            new_pos_map[v] = positive_map[k]
        self.positive_map = new_pos_map

    def get_fusion_op_iimage_format(self):
        return self.image_model_.get_fusion_op_iimage_format()

    def process_text(self, text):
        tokens, attention_mask = self.text_model_.text_tokenization(text)
        txt_encoded = self.text_model_.process(tokens)
        return (txt_encoded, tokens, attention_mask)

    def process_image_and_decode(self, text_features, image):
        img_encoded = self.image_model_.process(image)

        txt_encoded, tokens, attention_mask = text_features
        outputs = self.decoder_model_.process(
            img_encoded, txt_encoded, tokens, attention_mask
        )

        if isinstance(image, np.ndarray):
            img_h, img_w = image.shape[-2:]
        else:
            img_h, img_w = image.height, image.width
        return self.post_process(outputs, tokens[0], img_w, img_h)

    def post_process(
        self, decoder_outputs, input_ids, image_width, image_height, not_to_xyxy=False
    ):
        vacc_output_0 = decoder_outputs[0][:900, :195]
        vacc_output_1 = decoder_outputs[1][:900, :]
        res = torch.from_numpy(vacc_output_0.astype(np.float32))
        new_res = torch.full((*res.shape[:-1], 256), float("-inf"))
        new_res[..., : res.shape[-1]] = res
        logits = new_res.reshape(1, 900, 256)  # (nq, 256)
        boxes = torch.from_numpy(
            vacc_output_1.reshape(1, 900, 4).astype(np.float32)
        )  # (nq, 4)

        num_select = 300
        prob_to_token = logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        # (bs, 100, 256) @ (91, 256).T -> (bs, 100, 91)
        prob_to_label = prob_to_token @ pos_maps.T

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(
            prob.view(logits.shape[0], -1), num_select, dim=1
        )
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        if not_to_xyxy:
            boxes = boxes
        else:
            boxes = self.box_cxcywh_to_xyxy(boxes)

        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h = torch.tensor([image_height])
        img_w = torch.tensor([image_width])
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            [float(score), bbox.numpy(), self.category_dict[int(phrase)]]
            for score, bbox, phrase in zip(
                scores.squeeze(), boxes.squeeze(), labels.squeeze()
            )
            if score > self.threshold_
        ]
        return results

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def build_captions_and_token_span(self, cat_list, force_lowercase):
        """
        Return:
            captions: str
            cat2tokenspan: dict
                {
                    'dog': [[0, 2]],
                    ...
                }
        """

        cat2tokenspan = {}
        captions = ""
        for catname in cat_list:
            class_name = catname
            if force_lowercase:
                class_name = class_name.lower()
            if "/" in class_name:
                class_name_list: List = class_name.strip().split("/")
                class_name_list.append(class_name)
                class_name: str = random.choice(class_name_list)

            tokens_positive_i = []
            subnamelist = [i.strip() for i in class_name.strip().split(" ")]
            for subname in subnamelist:
                if len(subname) == 0:
                    continue
                if len(captions) > 0:
                    captions = captions + " "
                strat_idx = len(captions)
                end_idx = strat_idx + len(subname)
                tokens_positive_i.append([strat_idx, end_idx])
                captions = captions + subname

            if len(tokens_positive_i) > 0:
                captions = captions + " ."
                cat2tokenspan[class_name] = tokens_positive_i

        return captions, cat2tokenspan

    def create_positive_map_from_span(self, tokenized, token_span, max_text_len=256):
        """construct a map such that positive_map[i,j] = True iff box i is associated to token j
        Input:
            - tokenized:
                - input_ids: Tensor[1, ntokens]
                - attention_mask: Tensor[1, ntokens]
            - token_span: list with length num_boxes.
                - each item: [start_idx, end_idx]
        """
        positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
        for j, tok_list in enumerate(token_span):
            for beg, end in tok_list:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
                if beg_pos is None:
                    try:
                        beg_pos = tokenized.char_to_token(beg + 1)
                        if beg_pos is None:
                            beg_pos = tokenized.char_to_token(beg + 2)
                    except:
                        beg_pos = None
                if end_pos is None:
                    try:
                        end_pos = tokenized.char_to_token(end - 2)
                        if end_pos is None:
                            end_pos = tokenized.char_to_token(end - 3)
                    except:
                        end_pos = None
                if beg_pos is None or end_pos is None:
                    continue

                assert beg_pos is not None and end_pos is not None
                positive_map[j, beg_pos : end_pos + 1].fill_(1)

        return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

    def process(self, text, image):
        text_encoded = self.process_text(text)
        return self.process_image_and_decode(text_encoded, image)
