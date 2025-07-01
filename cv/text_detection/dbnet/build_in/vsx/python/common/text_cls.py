import numpy as np
from common.model_cv import ModelCV, vsx

class TextClassifier(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        label_list,
        batch_size=1,
        device_id=0,
        hw_config="",
    ):
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)
        self.label_list = label_list
        self.model_in_width = self.model_.input_shape[0][3]
        self.model_in_height = self.model_.input_shape[0][2]

    def process_impl(self, input):
        params = {  "rgb_letterbox_ext": [] }
        for vsx_image in input:
          radio = vsx_image.width / vsx_image.height
          resize_h = self.model_in_height
          resize_w = self.model_in_width if (self.model_in_height * radio > self.model_in_width) else self.model_in_height * radio
          resize_w = int(resize_w)
          right = 0 if (self.model_in_width - resize_w < 0) else self.model_in_width - resize_w 
          params["rgb_letterbox_ext"].append(
            #(resize_width , resize_height , top , bottom ,left, right)]
            ( int(resize_w) , resize_h, 0 , 0, 0, int(right))
          )
        outputs = self.stream_.run_sync(input, params)
        return self.post_process(outputs)

    def post_process(self, fp16_tensors):
        preds = np.array([vsx.as_numpy(out[0]) for out in fp16_tensors]).squeeze()
        if len(preds.shape) == 1:
            preds = preds.reshape(1, -1)
        label_list = self.label_list
        if label_list is None:
            label_list = {idx: idx for idx in range(preds.shape[-1])}

        pred_idxs = preds.argmax(axis=1)
        decode_out = [(label_list[idx], preds[i][idx])
                      for i, idx in enumerate(pred_idxs)]
        return decode_out
