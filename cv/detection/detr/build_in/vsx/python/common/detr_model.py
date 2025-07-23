from .model_cv import ModelCV, vsx
import numpy as np


class DetrModel(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        threshold=0.2,
        hw_config="",
    ) -> None:
        super().__init__(
            model_prefix,
            vdsp_config,
            batch_size,
            device_id,
            hw_config,
        )
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
        def np_softmax(x, axis=-1):
            """Compute softmax values for each sets of scores in x along axis."""
            e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return e_x / e_x.sum(axis=axis, keepdims=True)

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

        def unpad(x, dw, dh, r):
            x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
            b = [(x1 - dw) / r, (y1 - dh) / r, (x2 - dw) / r, (y2 - dh) / r]
            return np.stack(b, axis=-1)

        model_height, model_width = self.model_.input_shape[0][-2:]
        out_logits = vsx.as_numpy(fp16_tensors[0]).astype(np.float32)
        out_bbox = vsx.as_numpy(fp16_tensors[1]).astype(np.float32)

        prob = np_softmax(out_logits, -1)
        scores, labels = np_max(prob[..., :-1], axis=-1)

        boxes = box_cxcywh_to_xyxy(out_bbox)
        scale_fct = [[[model_width, model_height, model_width, model_height]]]
        boxes = boxes * scale_fct
        r = min(model_width / image_width, model_height / image_height)
        unpad_w = image_width * r
        unpad_h = image_height * r
        dw = (model_width - unpad_w) / 2
        dh = (model_height - unpad_h) / 2
        boxes = unpad(boxes, dw, dh, r).squeeze()

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
