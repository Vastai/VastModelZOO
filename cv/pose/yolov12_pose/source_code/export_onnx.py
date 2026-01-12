
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from ultralytics import YOLO

model = YOLO("/path/to/yolov12n-pose.pt")

# 在yolov12n-pose.pt同级目录生成yolov12n-pose.onnx
# onnx文件不包含后处理部分，输出有9个feature map
path = model.export(format="onnx")