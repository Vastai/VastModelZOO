# RepVGG onnx 导出
## official
```bash
git clone https://github.com/DingXiaoH/RepVGG.git
mv export.py RepVGG
cd RepVGG
python export.py pretrained/RepVGG-A0-train.pth save_dir/
```
## mmcls

mmcls框架参考 [mmclassification](https://github.com/open-mmlab/mmpretrain/tree/v0.23.2),可使用如下位置的pytorch2onnx.py或pytorch2torchscript.py转成相应的模型
```bash
cd mmclassification

## reparameterize model
python tools/convert_models/reparameterize_model.py \
    configs/repvgg/repvgg-A0_4xb64-coslr-120e_in1k.py \
    repvgg/pretrained/repvgg-A0_3rdparty_4xb64-coslr-120e_in1k_20210909-883ab98c.pth \
    repvgg/pretrained/repvgg-A0_deploy.pth
## export onnx
python tools/deployment/pytorch2onnx.py \
        --config configs/resnet/resnet50_b32x8_imagenet.py \
        --checkpoint weights/resnet50.pth \
        --output-file output/resnet50.onnx \
```
## ppcls

```bash
pip install PaddlePaddle==2.3.2  Paddle2ONNX==1.0.0

paddle2onnx  --model_dir /path/to/resnet_paddle_model/ \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file model.onnx \
            --enable_dev_version False \
            --opset_version 10
```
