
# 
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
from .model_cv import ModelCV, vsx
import numpy as np


class RtDetrModel(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        threshold=0.2,
        hw_config="",
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)
        self.threshold_ = threshold

    def set_threshold(self, threshold):
        self.threshold_ = threshold

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [
            self.post_process(output, input[i].width, input[i].height)
            for i, output in enumerate(outputs)
        ]

    def post_process(self, fp16_tensors, image_width, image_height):
        def np_sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            return s

        def np_max(x, axis=None, keepdims=False):
            max_val = np.max(x, axis=axis, keepdims=keepdims)
            max_idx = np.argmax(x, axis=axis)
            if keepdims:
                max_idx = np.expand_dims(max_idx, axis)
            return max_val, max_idx

        def box_cxcywh_to_xyxy(x):
            x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return np.stack(b, axis=-1)

        out_logits = vsx.as_numpy(fp16_tensors[0]).astype(np.float32)
        out_bbox = vsx.as_numpy(fp16_tensors[1]).astype(np.float32)

        prob = np_sigmoid(out_logits)
        scores, labels = np_max(prob[..., :-1], axis=-1)

        boxes = box_cxcywh_to_xyxy(out_bbox)
        scale_fct = [[[image_width, image_height, image_width, image_height]]]
        boxes = boxes * scale_fct
        boxes = boxes.squeeze()

        # indies = scores.argsort()[::-1]
        # scores = scores[indies]
        # boxes = boxes[indies]
        # labels = labels[indies]

        data_count = len(scores)
        result_np = np.zeros((data_count, 6), dtype=np.float32) - 1
        n = 0
        for i in range(data_count):
            score = scores[i]
            if score >= self.threshold_:
                box = boxes[i]
                result_np[n][0] = labels[i]
                result_np[n][1] = score
                result_np[n][2] = box[0]
                result_np[n][3] = box[1]
                result_np[n][4] = box[2] - box[0]
                result_np[n][5] = box[3] - box[1]
                n += 1
        return result_np
