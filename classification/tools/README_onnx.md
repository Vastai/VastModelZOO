# x2onnx/torchscript

下面以resnet50为例，展示各种来源的预训练模型转成onnx或torchscript

## torchvision、timm模型转成onnx、torchscript

```bash
pip install thop torch=1.8.0 torchvision==0.9.0 timm==0.6.5 onnx==1.10.0   

python cls_mode_hub.py \
        --model_library timm \
        --model_name resnet50 \
        --save_dir output/ \
        --pretrained_weights weights/resnet50.pth \
        --convert_mode pt \
```

## paddle模型转成onnx
```bash
pip install PaddlePaddle==2.3.2  Paddle2ONNX==1.0.0

paddle2onnx  --model_dir /path/to/resnet_paddle_model/ \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file model.onnx \
            --enable_dev_version False \
            --opset_version 10
```

## mmcls模型转成onnx、torchscript

mmcls框架参考[mmclassification](https://github.com/open-mmlab/mmclassification)，可使用如下位置的pytorch2onnx.py或pytorch2torchscript.py转成相应的模型

```bash
cd mmclassification

python tools/deployment/pytorch2onnx.py \
        --config configs/resnet/resnet50_b32x8_imagenet.py \
        --checkpoint weights/resnet50.pth \
        --output-file output/resnet50.onnx \
```


## 其它

一般可在原模型推理或评估脚本内，加载权重后的适当位置添加以下代码，完成模型转为onnx或torchscript：

```python
model = net.eval()

input_shape = (1, 3, 224, 224)
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)

scripted_model = torch.jit.trace(model, input_data).eval()
scripted_model.save(args.checkpoint.replace(".pth", ".torchscript.pt"))
scripted_model = torch.jit.load(args.checkpoint.replace(".pth", ".torchscript.pt"))

import onnx
torch.onnx.export(model, input_data, args.checkpoint.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=10)
shape_dict = {"input": input_shape}
onnx_model = onnx.load(args.checkpoint.replace(".pth", ".onnx"))
```

## onnx-simplifier
如有必要可对onnx模型进行简化，去除胶水节点，以及做一些图优化，得到一个简洁明了的模型图：

```bash
pip install onnx-simplifier

python3 -m onnxsim input_onnx_model output_onnx_model
```