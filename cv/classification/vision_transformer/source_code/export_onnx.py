# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from transformers import ViTForImageClassification
import torch

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
inputs = torch.randn(1, 3, 224, 224)
onnx_model = "vit_base.onnx"
model.eval()
torch.onnx.export(model, inputs, onnx_model, opset_version=12)


# tested in
# torch 1.13.1