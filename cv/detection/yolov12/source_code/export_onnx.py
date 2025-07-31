
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from ultralytics import YOLO

model = YOLO('models/pt_model/yolov12s.pt')
model.export(format="onnx", opset=17, simplify=True)  # or format="onnx"