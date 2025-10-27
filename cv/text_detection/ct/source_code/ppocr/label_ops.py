# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import json


class CTLabelEncode(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        label = data['label']

        label = json.loads(label)
        nBox = len(label)
        boxes, txts = [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            box = np.array(box)

            boxes.append(box)
            txt = label[bno]['transcription']
            txts.append(txt)

        if len(boxes) == 0:
            return None

        data['polys'] = boxes
        data['texts'] = txts
        return data