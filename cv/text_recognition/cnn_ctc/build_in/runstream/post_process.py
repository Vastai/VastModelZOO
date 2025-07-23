# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
'@Author :        melodylu
'@Email :   algorithm@vastaitech.com
'@Time  :     2025/07/23 18:01:22
'''

import abc
import re

import torch
import torch.nn.functional as F


class BaseConverter(object):

    def __init__(self, character):
        self.character = list(character)
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    @abc.abstractmethod
    def train_encode(self, *args, **kwargs):
        '''encode text in train phase'''

    @abc.abstractmethod
    def test_encode(self, *args, **kwargs):
        '''encode text in test phase'''

    @abc.abstractmethod
    def decode(self, *args, **kwargs):
        '''decode label to text in train and test phase'''

class FCConverter(BaseConverter):

    def __init__(self, character, batch_max_length=25):

        list_token = ['[s]']
        ignore_token = ['[ignore]']
        list_character = list(character)
        self.batch_max_length = batch_max_length + 1
        super(FCConverter, self).__init__(character=list_token + list_character + ignore_token)
        self.ignore_index = self.dict[ignore_token[0]]

    def encode(self, text):
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.ignore_index)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        batch_text_input = batch_text
        batch_text_target = batch_text

        return batch_text_input, torch.IntTensor(length), batch_text_target

    def train_encode(self, text):
        return self.encode(text)

    def test_encode(self, text):
        return self.encode(text)

    def decode(self, text_index):
        texts = []
        batch_size = text_index.shape[0]
        for index in range(batch_size):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            text = text[:text.find('[s]')]
            texts.append(text)

        return texts


cfg = dict(
        sensitive=False,
        character='0123456789abcdefghijklmnopqrstuvwxyz',
    )

converter = FCConverter(cfg['character'])

def postprocess(preds, cfg=cfg):
    if cfg is not None:
        sensitive = cfg.get('sensitive', True)
        character = cfg.get('character', '')
    else:
        sensitive = True
        character = ''

    probs = F.softmax(preds, dim=2)
    max_probs, indexes = probs.max(dim=2)
    preds_str = []
    preds_prob = []
    for i, pstr in enumerate(converter.decode(indexes)):
        str_len = len(pstr)
        if str_len == 0:
            prob = 0
        else:
            prob = max_probs[i, :str_len].cumprod(dim=0)[-1]
        preds_prob.append(prob)
        if not sensitive:
            pstr = pstr.lower()

        if character:
            pstr = re.sub('[^{}]'.format(character), '', pstr)

        preds_str.append(pstr)

    return preds_str, preds_prob
