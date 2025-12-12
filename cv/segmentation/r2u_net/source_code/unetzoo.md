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
1. 根据具体模型，修改编译配置
    - [official_r2u_net.yaml](../build_in/build/official_r2u_net.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd r2u_net
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_r2u_net.yaml
    ```

### step.4 模型推理

- 参考：[vsx_inference.py](../build_in/vsx/python/vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/vsx_inference.py \
        --image_dir  /path/to/dsb2018_256_val/images \
        --model_prefix_path deploy_weights/official_r2u_net_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/unetzoo-r2u_net-vdsp_params.json \
        --mask_dir /path/to/dsb2018_256_val/masks \
        --save_dir ./infer_output \
        --device 0
    ```

    ```
    # int8
    mean DIC: 85.449, mean IoU: 76.686

    # fp16
    mean DIC: 85.867, mean IoU: 77.237
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[unetzoo-r2u_net-vdsp_params.json](../build_in/vdsp_params/unetzoo-r2u_net-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_r2u_net_fp16/mod \
    --vdsp_params ../build_in/vdsp_params/unetzoo-r2u_net-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,96,96]
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式（注意配置图片后缀为`.png`）：
    ```bash
    python ../build_in/vdsp_params/image2npz.py \
        --dataset_path /path/to/dsb2018/dsb2018_256_val/images \
        --target_path  /path/to/dsb2018/dsb2018_256_val/images_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果
    ```bash
    vamp -m deploy_weights/official_r2u_net_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/unetzoo-r2u_net-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,96,96] \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)，
    ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
        --src_dir /path/to/dsb2018/dsb2018_256_val/images \
        --gt_dir /path/to/dsb2018/dsb2018_256_val/masks \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output \
        --input_shape 96 96 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```

### Tips
- 注意输入图像需设置为`BGR`，否则精度有损失
