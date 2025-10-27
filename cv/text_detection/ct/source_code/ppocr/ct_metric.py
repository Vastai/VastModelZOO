# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from scipy import io
import numpy as np

from ppocr.Deteval import combine_results, get_score_C


class CTMetric(object):
    def __init__(self, delimiter='\t', **kwargs):
        self.delimiter = delimiter
        # self.main_indicator = main_indicator
        self.reset()

    def reset(self):
        self.results = []  # clear results

    def __call__(self, preds, batch, **kwargs):
        # NOTE: only support bs=1 now, as the label length of different sample is Unequal 
        assert len(
            preds) == 1, "CentripetalText test now only suuport batch_size=1."
        label = batch[0]
        text = batch[1]
        pred = preds['points']
        result = get_score_C(label, text, pred)

        self.results.append(result)

    def get_metric(self):
        """
        Input format: y0,x0, ..... yn,xn. Each detection is separated by the end of line token ('\n')'
        """
        metrics = combine_results(self.results, rec_flag=False)
        self.reset()
        return metrics
