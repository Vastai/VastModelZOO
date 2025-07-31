# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from PIL import Image
from yolo import YOLO, YOLO_ONNX

if __name__ == "__main__":

    simplify        = True
    onnx_save_path  = "model_data/models.onnx"
    
    # 注意修改YOLO类内的相关参数
    yolo = YOLO()
    yolo.convert_to_onnx(simplify, onnx_save_path)

    # onnx run
    # 注意修改YOLO_ONNX类内的相关参数
    yolo_onnx = YOLO_ONNX()
    image = Image.open("/path/to/img/street.jpg")
    r_image = yolo_onnx.detect_image(image)
    r_image.save("draw.png")