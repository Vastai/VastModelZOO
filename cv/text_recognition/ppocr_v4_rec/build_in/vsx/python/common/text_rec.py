# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from .model_cv import ModelCV, vsx
import numpy as np


class BaseRecLabelDecode(object):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if "arabic" in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ""
        for c in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)

        return "".join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token
            
            print(text_index[batch_idx][selection])
            char_list = [
                self.character[text_id] for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


class TextRecognizer(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        label_path=None,
        batch_size=1,
        device_id=0,
        hw_config="",
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)
        self.decoder = CTCLabelDecode(label_path, True)
        self.model_in_width = self.model_.input_shape[0][3]
        self.model_in_height = self.model_.input_shape[0][2]

    def set_threshold(self, threshold):
        self.threshold_ = threshold

    def process_impl(self, input):
        params = {"rgb_letterbox_ext": []}
        for vsx_image in input:
            radio = vsx_image.width / vsx_image.height
            print(f"vsx_image.width:{vsx_image.width} vsx_image.height:{vsx_image.height} radio:{radio}")
            print(radio)
            resize_h = self.model_in_height
            resize_w = (
                self.model_in_width
                if (self.model_in_height * radio > self.model_in_width)
                else self.model_in_height * radio
            )
            resize_w = int(resize_w)
            right = (
                0
                if (self.model_in_width - resize_w < 0)
                else self.model_in_width - resize_w
            )
            params["rgb_letterbox_ext"].append(
                # (resize_width , resize_height , top , bottom ,left, right)]
                (int(resize_w), resize_h, 0, 0, 0, int(right))
            )
        print(params["rgb_letterbox_ext"])
        print(vsx.as_numpy(input[0]))
        outputs = self.stream_.run_sync(input, params)
        return [
            self.post_process(output, input[i].width, input[i].height)
            for i, output in enumerate(outputs)
        ]

    def post_process(self, fp16_tensors, image_width, image_height):
        host_tsrs = vsx.as_numpy(fp16_tensors[0])
        print(host_tsrs)
        return self.decoder(np.expand_dims(host_tsrs, axis=0))
