## VainF版本

### step.1 获取预训练模型

```
link: https://github.com/DrSleep/refinenet-pytorch
branch: master
commit: 8f25c076016e61a835551493aae303e81cf36c53

link: https://github.com/DrSleep/light-weight-refinenet
branch: master
commit: 538fe8b39327d8343763b859daf7b9d03a05396e
```

一般在原始仓库内进行模型转为onnx或torchscript。在原仓库test或val脚本内，定义模型和加载训练权重后，添加以下脚本可实现：

- refinenet仓库，如[VOC.ipynb](https://github.com/DrSleep/refinenet-pytorch/blob/master/examples/notebooks/VOC.ipynb)
- light-weight-refinenet仓库，如[VOC.ipynb](https://github.com/DrSleep/light-weight-refinenet/blob/master/examples/notebooks/VOC.ipynb)

```python
args.weights_test = "path/to/trained/weight.pth"
model = model.eval()
input_shape = (1, 3, 500, 500)
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()
scripted_model.save(args.weights_test.replace(".pth", ".torchscript.pt"))
scripted_model = torch.jit.load(args.weights_test.replace(".pth", ".torchscript.pt"))

import onnx
torch.onnx.export(model, input_data, args.weights_test.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
            # dynamic_axes= {
            #                 "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
            #                 "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}
            #                 }
)
shape_dict = {"input": input_shape}
onnx_model = onnx.load(args.weights_test.replace(".pth", ".onnx"))
```


### step.2 准备数据集
- 下载[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)数据集，解压
- 使用[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，提取val图像数据集和转换为npz格式


### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[config.yaml](../vacc_code/build/config.yaml)
3. 执行转换

   ```bash
    vamc build ../vacc_code/build/config.yaml
   ```

### step.4 benchmark
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`，注意只转换`VOC2012/ImageSets/Segmentation/val.txt`对应的验证集图像（配置相应路径）：
    ```bash
    python ../vacc_code/vdsp_params/image2npz.py \
    --dataset_path VOC2012/JPEGImages \
    --target_path  VOC2012/JPEGImages_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[drsleep-refinenet_resnet101-vdsp_params.json](../vacc_code/vdsp_params/drsleep-refinenet_resnet101-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/refinenet_resnet101-int8-kl_divergence-3_500_500-vacc/refinenet_resnet101 \
    --vdsp_params ../vacc_code/vdsp_params/drsleep-refinenet_resnet101-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,500,500]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/refinenet_resnet101-int8-kl_divergence-3_500_500-vacc/refinenet_resnet101 \
    --vdsp_params vacc_code/vdsp_params/drsleep-refinenet_resnet101-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,500,500] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --src_dir VOC2012/JPEGImages_val \
    --gt_dir VOC2012/SegmentationClass \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 500 500 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


### Tips
