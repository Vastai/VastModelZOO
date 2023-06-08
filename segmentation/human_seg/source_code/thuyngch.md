
## thuyngch版本

### step.1 获取预训练模型
```
link: https://github.com/thuyngch/Human-Segmentation-PyTorch
branch: master
commit: b15baef04e7b628c01a4526de5e14f9524f18da6
```

一般在原始仓库内进行模型转为onnx或torchscript。在原仓库test或val脚本内，如[inference_video.py#L100](https://github.com/thuyngch/Human-Segmentation-PyTorch/blob/master/inference_video.py#L100)，定义模型和加载训练权重后，添加以下脚本可实现：

```python
args.weights_test = "path/to/trained/weight.pth"
model = self.model.eval()
input_shape = (1, 3, 320, 320)
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()
scripted_model.save(args.weights_test.replace(".pth", ".torchscript.pt"))
scripted_model = torch.jit.load(args.weights_test.replace(".pth", ".torchscript.pt"))

import onnx
torch.onnx.export(model, input_data, args.weights_test.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11)
shape_dict = {"input": input_shape}
onnx_model = onnx.load(args.weights_test.replace(".pth", ".onnx"))
```


### step.2 准备数据集
- 下载[Supervisely Person](https://ecosystem.supervise.ly/projects/persons/)数据集，解压
- 按[link](https://blog.csdn.net/u011622208/article/details/108535943)整理转换数据集

### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[config.yaml](../vacc_code/build/config.yaml)
3. 执行转换

   ```bash
    vamc build ../vacc_code/build/config.yaml
   ```

### step.4 benchmark
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../vacc_code/vdsp_params/image2npz.py \
    --dataset_path Supervisely_Person_Dataset/src \
    --target_path  Supervisely_Person_Dataset/src_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[thuyngch-deeplabv3plus_resnet18-vdsp_params.json](../vacc_code/vdsp_params/thuyngch-deeplabv3plus_resnet18-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/deeplabv3plus_resnet18-int8-kl_divergence-3_320_320-vacc/deeplabv3plus_resnet18 \
    --vdsp_params ../vacc_code/vdsp_params/thuyngch-deeplabv3plus_resnet18-vdsp_params.json \
    -i 2 p 2 -b 1
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/deeplabv3plus_resnet18-int8-kl_divergence-3_320_320-vacc/deeplabv3plus_resnet18 \
    --vdsp_params vacc_code/vdsp_params/thuyngch-deeplabv3plus_resnet18-vdsp_params.json \
    -i 2 p 2 -b 1 \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --src_dir Supervisely_Person_Dataset/src \
    --gt_dir Supervisely_Person_Dataset/mask \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 320 320 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


### Tips
- 源仓库用到了timm，需要限定版本0.1.12
- 源仓库提供的训练权重链接，已无法下载。基于给定数据集自己训练了5个模型，精度有差异
- 2023/03/20，当前只支持deeplabv3plus_resnet18，unet_resnet18和unet_mobilenetv2三个模型，其中deeplabv3plus_resnet18，vacc fp16 run error。其他模型bulid会报错。
