# Mobilenet_v3 
## timm模型导出onnx

    ```bash
    pip install timm==0.6.5
    python ../common/utils/export_timm_torchvision_model.py --model_library timm  --model_name mobilenetv3_rw --save_dir ./onnx  --size 224 --pretrained_weights xxx.pth
    ```

## torchvision模型导出onnx
    ```bash
    python ../common/utils/export_timm_torchvision_model.py --model_library torchvision  --model_name mobilenet_v3_small --save_dir ./onnx  --size 224 --pretrained_weights xxx.pth
    ```

## paddle模型导出onnx
    ```bash
    pip install PaddlePaddle==2.3.2  Paddle2ONNX==1.0.0

    paddle2onnx  --model_dir /path/to/paddle_model/ \
                --model_filename model.pdmodel \
                --params_filename model.pdiparams \
                --save_file model.onnx \
                --enable_dev_version False \
                --opset_version 10
    ```
## showlo模型导出onnx
    ```bash
    git clone https://github.com/ShowLo/MobileNetV3.git
    mv source_code/export_onnx.py MobileNetV3 & cd MobileNetV3
    python export_onnx.py
    ```