
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :    zpwen
@Email  :    algorithm@vastaitech.com
@Time   :    2025/06/12 16:17:38
'''

from ultralytics import YOLO

model = YOLO("./yolov11/official/yolo11s.pt")

# 在yolo11s.pt同级目录生成yolo11s.onnx
path = model.export(format="onnx")