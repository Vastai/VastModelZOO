# resnet
下面以resnet50为例，展示各种来源的预训练模型转成onnx或torchscript

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
mmcls框架参考 [mmclassification](https://github.com/open-mmlab/mmclassification),可使用如下位置的pytorch2onnx.py或pytorch2torchscript.py转成相应的模型
```bash
cd mmclassification

python tools/deployment/pytorch2onnx.py \
        --config configs/resnet/resnet50_b32x8_imagenet.py \
        --checkpoint weights/resnet50.pth \
        --output-file output/resnet50.onnx \
```

## torchvision、timm模型转成onnx或torchscript

基于[cls_mode_hub.py](../../common/utils/cls_mode_hub.py)，进行转换

```bash
python cls_mode_hub.py \
        --model_library timm \
        --model_name resnet50 \
        --save_dir output/ \
        --pretrained_weights weights/resnet50.pth \
        --convert_mode pt \
```

## oneflow

oneflow库提供了oneflow_onnx工具将原模型转为onnx格式，可以参考以下代码完成转换。默认转成的onnx模型输入输出name不统一，可以利用脚本完成输入输出name的转换，完整代码见[oneflow2onnx.py](./oneflow2onnx.py)

```python
class ResNetGraph(nn.Graph):
    def __init__(self, eager_model):
        super().__init__()
        self.model = eager_model

    def build(self, x):
        return self.model(x)

resnet_model = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
save_dir = './onnx/'
for m in resnet_model:
    model = ModelCreator.create_model(m, pretrained=True)
    
for m in resnet_model:
    MODEL_PARAMS = 'resnet/' + m
    params = flow.load(MODEL_PARAMS)
    model = eval(m)()
    model.load_state_dict(params)

    # 将模型设置为 eval 模式
    model.eval()

    resnet_graph = ResNetGraph(model)
    # 构建出静态图模型
    resnet_graph._compile(flow.randn(1, 3, 224, 224))

    # 导出为 ONNX 模型并进行检查
    convert_to_onnx_and_check(resnet_graph, 
                            onnx_model_path=save_dir + m + '.onnx', 
                            print_outlier=True,
                            dynamic_batch_size=True)
    
onnx_files = glob.glob(save_dir + '/*.onnx')
for onnx_file in onnx_files:
    rename_onnx_node(onnx_file, [' ', ' '], ['input', 'output'])
```