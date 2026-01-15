
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from ultralytics import YOLO

model = YOLO("./yolov11/official/yolo11n-pose.pt")

# 在yolo11n-pose.pt同级目录生成yolo11n-pose.onnx
# onnx文件不包含后处理部分，输出有9个feature map
path = model.export(format="onnx")