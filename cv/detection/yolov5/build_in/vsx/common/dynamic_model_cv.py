from .dynamic_model_base import DynamicModelBase, vsx
import numpy as np
from typing import Union, List


def cv_rgb_image_to_vastai(image_cv, device_id):
    assert len(image_cv.shape) >= 2
    h, w = image_cv.shape[:2]
    if len(image_cv.shape) == 3:
        return vsx.create_image(
            image_cv, vsx.ImageFormat.BGR_INTERLEAVE, w, h, device_id
        )
    elif len(image_cv.shape) == 2:
        return vsx.create_image(image_cv, vsx.ImageFormat.GRAY, w, h, device_id)
    else:
        raise Exception("unsupported ndarray shape:", image_cv.shape)


class DynamicModelCV(DynamicModelBase):
    def __init__(
        self, module_info, vdsp_config, max_input_size, batch_size=1, device_id=0
    ) -> None:
        super().__init__(
            module_info, vdsp_config, max_input_size, batch_size, device_id
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
        else:
            if isinstance(input, np.ndarray):
                return self.process(cv_rgb_image_to_vastai(input, self.device_id_))
            else:
                return self.process_impl([input])[0]

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height, width = input_shape[-2:]
        dummy = np.zeros((height, width, 3), dtype=dtype)
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            vacc_dummy = vsx.create_image(
                dummy, vsx.ImageFormat.BGR_INTERLEAVE, width, height, self.device_id_
            )
            return [vacc_dummy] * batch_size

    def set_model_input_shape(self, model_input_shape):
        self.model_input_shape_ = model_input_shape

    def process_impl(self, inputs):
        assert len(inputs) == len(self.model_input_shape_)
        input_shapes = [[shape] for shape in self.model_input_shape_]
        outputs = self.stream_.run_sync(
            inputs, {"dynamic_model_input_shapes": input_shapes}
        )
        return [vsx.as_numpy(out[0]) for out in outputs]
