# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from ultralytics import YOLO

model = YOLO("/path/to/yolo12s-pose.yaml")
results = model.train(data="coco-pose.yaml", epochs=100, imgsz=640)
