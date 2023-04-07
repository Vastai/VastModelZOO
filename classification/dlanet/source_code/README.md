# dlanet

## ucbdrive模型导出onnx
可自定义模型导出
```bash
git clone https://github.com/ucbdrive/dla.git
mv export_onnx.py dla & cd dla
python export_onnx.py
```

## paddle模型导出onnx
```bash
pip install PaddlePaddle==2.3.2  Paddle2ONNX==1.0.0

paddle2onnx  --model_dir /path/to/dlanet_paddle_model/ \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file model.onnx \
            --enable_dev_version False \
            --opset_version 10
```

## timm模型导出onnx或torchscript

```bash
pip install timm==0.6.5
python ../common/utils/export_timm_torchvision_model.py --model_library timm  --model_name dla34 --save_dir ./onnx  --size 224 --pretrained_weights xxx.pth
```
