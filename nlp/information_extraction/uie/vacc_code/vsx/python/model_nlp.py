# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :          Steven
@Email : xinghe@vastaitech.com
@Time  : 	2025/06/20 16:19:31
'''



from model_base import ModelBase, vsx
import numpy as np
from typing import Union, List


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
