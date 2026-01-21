
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from ultralytics import YOLO

model = YOLO("./yolov11/official/yolo11n-cls.pt")

# 在yolo11n-cls.pt同级目录生成yolo11n-cls.onnx
model.export(format="onnx", imgsz=256)