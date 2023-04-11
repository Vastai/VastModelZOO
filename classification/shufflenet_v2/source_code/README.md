# shufflenet_v2
## torchvision导出onnx

    ```bash
    python ../common/utils/export_timm_torchvision_model.py --model_library torchvision  --model_name shufflenet_v2 --save_dir ./onnx  --size 224 --pretrained_weights xxx.pth
    ```

## ppcls导出onnx

    ```bash
    pip install PaddlePaddle==2.3.2  Paddle2ONNX==1.0.0

    paddle2onnx  --model_dir /path/to/resnet_paddle_model/ \
                --model_filename model.pdmodel \
                --params_filename model.pdiparams \
                --save_file model.onnx \
                --enable_dev_version False \
                --opset_version 10
    ```
## megvii模型导出onnx
    ```bash
    git clone https://github.com/megvii-model/ShuffleNet-Series.git
    mv source_code/export_onnx.py ShuffleNet-Series/ShuffleNetV2
    cd ShuffleNet-Series/ShuffleNetV2
    python export_onnx.py
    ```