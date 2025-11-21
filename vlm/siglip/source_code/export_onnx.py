# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from transformers import SiglipVisionModel, SiglipProcessor
from PIL import Image
import torch
from torch import nn

class BinaryClassifier(nn.Module):
       def __init__(self):
              super().__init__()
              self.vision_model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
              # 获取 vision model 的输出维度
              num_classes = 1
              num_features = self.vision_model.config.hidden_size

              self.classifier = nn.Sequential(
              nn.Linear(num_features, 256),
                     nn.ReLU(),
                     nn.Dropout(0.2),
                     nn.Linear(256, num_classes),
                     nn.Sigmoid(),
              )
              
              self.processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")
       
       def forward(self, pixel_values):
              # image = Image.open(img_path).convert("RGB")
              # inputs = self.processor(images=image, return_tensors="pt")
              # # pixel_values = inputs.pixel_values.squeeze()
              # pixel_values = inputs.pixel_values
              outputs = self.vision_model(pixel_values)
              pooled_output = outputs.pooler_output
              logits = self.classifier(pooled_output)
              return logits
       
binaryCl_model = BinaryClassifier()
binaryCl_model.eval()

# export onnx model
dummpy_pixel_values = torch.normal(0, 1, size=(1,3,384,384))
onnx_path = "siglip-instruct.onnx"

torch.onnx.export(
       binaryCl_model,
       (dummpy_pixel_values),
       onnx_path,
       input_names = ['pixel_values'],
       output_name = ['logits'],
       opset_version=14,
       verbose=False,
       export_params=True,
       do_constant_folding=True
       
       
)
print("export siglip-instruct onnx model OK")