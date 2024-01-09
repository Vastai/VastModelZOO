## UnetZOO版本

```
link: https://github.com/Andy-zhujunwen/UNET-ZOO
branch: master
commit: b526ce5dc2bef53249506883b92feb15f4f89bbb
```

### step.1 获取预训练模型

一般在原始仓库内进行模型转为onnx或torchscript。在原仓库test或val脚本内[main.py#L130](https://github.com/Andy-zhujunwen/UNET-ZOO/blob/master/main.py#L130)，定义模型和加载训练权重后，添加以下脚本可实现：

```python
args.weights_test = "path/to/trained/weight.pth"
input_shape = (1, 3, 96, 96)
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
- 下载[DSB2018](https://github.com/sunalbert/DSB2018)数据集，解压


### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[config.yaml](../vacc_code/build/config.yaml)
3. 执行转换

   ```bash
   vamc build ../vacc_code/build/config.yaml
   ```

### step.4 benchmark
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式（注意配置图片后缀为`.png`）：
    ```bash
    python ../vacc_code/vdsp_params/image2npz.py \
    --dataset_path dsb2018/dsb2018_256_val/images \
    --target_path  dsb2018/dsb2018_256_val/images_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[unetzoo-r2u_net-vdsp_params.json](../vacc_code/vdsp_params/unetzoo-r2u_net-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/r2u_net-int8-kl_divergence-3_96_96-vacc/r2u_net \
    --vdsp_params ../vacc_code/vdsp_params/unetzoo-r2u_net-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,96,96]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/r2u_net-int8-kl_divergence-3_96_96-vacc/r2u_net \
    --vdsp_params vacc_code/vdsp_params/unetzoo-r2u_net-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,96,96] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --src_dir dsb2018/dsb2018_256_val/images \
    --gt_dir dsb2018/dsb2018_256_val/masks \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir deploy_weights/r2u_net-int8-kl_divergence-3_96_96-debug-result \
    --input_shape 96 96 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


### Tips

- 注意输入图像需设置为`BGR`，否则精度有损失