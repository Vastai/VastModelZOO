
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from ultralytics import YOLO

model = YOLO("./yolov11/official/yolo11n-obb.pt")

# 在yolo11n-obb.pt同级目录生成yolo11n-obb.onnx
path = model.export(format="onnx")