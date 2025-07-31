# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from ultralytics import YOLOv10


# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10('yolov10x.pt')

# predict，这里可以随意指定一张图
aa = model.predict('/path/to/bus.jpg')
aa[0].save(filename="result.jpg")

# export onnx
model.export(format="onnx", imgsz=640, simplify=True)

# cut onnx
import onnx
onnx.utils.extract_model('./yolov10x.onnx', './yolov10x-sim_cut.onnx', ['images'], ['/model.23/Concat_output_0','/model.23/Concat_1_output_0','/model.23/Concat_2_output_0'])