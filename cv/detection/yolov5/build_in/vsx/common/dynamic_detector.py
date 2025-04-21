from .dynamic_model_cv import DynamicModelCV, vsx
import numpy as np


class DynamicDetector(DynamicModelCV):
    def __init__(
        self,
        module_info,
        vdsp_config,
        max_input_size,
        batch_size=1,
        device_id=0,
        threshold=0.2,
    ) -> None:
        super().__init__(
            module_info, vdsp_config, max_input_size, batch_size, device_id
        )
        self.threshold_ = threshold

    def set_threshold(self, threshold):
        self.threshold_ = threshold

    def process_impl(self, inputs):
        assert len(inputs) == len(self.model_input_shape_)
        input_shapes = [[shape] for shape in self.model_input_shape_]
        outputs = self.stream_.run_sync(
            inputs, {"dynamic_model_input_shapes": input_shapes}
        )
        return [
            self.post_process(
                output, inputs[i].width, inputs[i].height, self.model_input_shape_[i]
            )
            for i, output in enumerate(outputs)
        ]

    def post_process(self, out_tensors, image_width, image_height, model_input_shape):
        data_count = out_tensors[0].size
        result_np = np.zeros((data_count, 6), dtype=np.float32) - 1
        # check tensor size validation
        assert (
            out_tensors[0].size == out_tensors[1].size
            and out_tensors[1].size * 4 == out_tensors[2].size
        ), f"Output tensor size error, sizes are:{out_tensors[0].size},{out_tensors[1].size},{out_tensors[2].size}"
        class_data = vsx.as_numpy(out_tensors[0]).squeeze()
        score_data = vsx.as_numpy(out_tensors[1]).squeeze()
        bbox_data = vsx.as_numpy(out_tensors[2]).squeeze()

        model_height, model_width = model_input_shape[-2:]

        r = min(model_width / image_width, model_height / image_height)
        unpad_w = image_width * r
        unpad_h = image_height * r
        dw = (model_width - unpad_w) / 2
        dh = (model_height - unpad_h) / 2

        for i in range(data_count):
            category = int(class_data[i])
            if category < 0:
                break
            score = score_data[i]
            if score > self.threshold_:
                bbox_xmin = (bbox_data[i][0] - dw) / r
                bbox_ymin = (bbox_data[i][1] - dh) / r
                bbox_xmax = (bbox_data[i][2] - dw) / r
                bbox_ymax = (bbox_data[i][3] - dh) / r
                bbox_width = bbox_xmax - bbox_xmin
                bbox_height = bbox_ymax - bbox_ymin
                result_np[i][0] = category
                result_np[i][1] = score
                result_np[i][2] = bbox_xmin
                result_np[i][3] = bbox_ymin
                result_np[i][4] = bbox_width
                result_np[i][5] = bbox_height
        return result_np
