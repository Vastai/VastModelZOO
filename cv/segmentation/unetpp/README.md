## UNetPP

- Technical Report
    - [UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation](https://arxiv.org/abs/1912.05074)


## Code Source
```
link: https://github.com/Andy-zhujunwen/UNET-ZOO
branch: master
commit: b526ce5dc2bef53249506883b92feb15f4f89bbb
```

## Model Arch

<div  align="center">
<img src="../../../images/cv/segmentation/unetpp/arch.png">
</div>

### pre-processing

UNetPP网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至一定尺寸(96)，然后对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(scale_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],),
]
```

### post-processing

UNetPP算法的后处理即是对网络输出的heatmap进行逐像素判断，比如一共20个类别，则网络会输出21个通道(20class+background)尺寸等于原图大小的heatmap，然后逐像素判断哪个通道数值大，就表示当前像素点所代表的类别为当前通道对应的类别

### backbone

可灵活使用各种分类网络作为骨架作特征提取。UNetPP解决了不同数据量、不同场景应用对网络深度的要求。通过短连接和上下采样等操作，间接融合了多个不同层次的特征，而非简单的Encoder与Decoder同层级特征的简单拼接。正因为此，Decoder可以感知不同感受野下大小不一的object。
- 在UNetPP中引入了一个内置的深度可变的UNet集合，可为不同大小的对象提供改进的分割性能，这是对固定深度UNet的改进。
- 重新设计了UNetPP中的跳接，从而在解码器中实现了灵活的特征融合，这是对UNet中仅需要融合相同比例特征图的限制性跳接的一种改进。
- 设计了一种方案来剪枝经过训练的UNetPP，在保持其性能的同时加快其推理速度。
- 同时训练嵌入在UNetPP体系结构中的多深度U-Net可以激发组成UNet之间的协作学习，与单独训练具有相同体系结构的隔离UNet相比，可以带来更好的性能。
- 展示了UNetPP对多个主干编码器的可扩展性，并进一步将其应用于包括CT、MRI和电子显微镜在内的各种医学成像模式。

### common

- deep supervision

## Model Info

### 模型精度

|Models|Code Source|FLOPs(G)|Params(M)|MIoU|Shapes|
|:-:|:-:|:-:|:-:|:-:|:-:|
|UnetPP|[UnetZoo](https://github.com/Andy-zhujunwen/UNET-ZOO)|10.897|9.163|84.164|3×96×96|
|UnetPP|vacc fp16|-|-|83.179|3×96×96|
|UnetPP|vacc int8 kl_divergence|-|-|83.052|3×96×96|


### 测评数据集说明

[DSB2018](https://github.com/sunalbert/DSB2018)数据集，是显微镜下细胞图像中的细胞核分割数据，有细胞核和背景两个类别。

<div  align="center">
<img src="../../../images/dataset/dsb2018.png" width="70%" height="70%">
</div>


### 指标说明
- IoU并交比：两个区域重叠的部分除以两个区域的集合部分，取值TP/(TP+FN+FP)
- MIoU平均并交比：分割图像一般都有好几个类别，把每个分类得出的分数进行平均得到mean IoU，也就是mIoU，其是各种基准数据集最常用的标准之一，绝大数的图像语义分割论文中模型评估比较都以此作为主要评估指标。


## Build_In Deploy

### step.1 模型准备
1. 下载模型权重
    ```
    link: https://github.com/Andy-zhujunwen/UNET-ZOO
    branch: master
    commit: b526ce5dc2bef53249506883b92feb15f4f89bbb
    ```

2. 模型导出

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
1. 根据具体模型修改模型转换配置文件
    - [official_unetpp.yaml](./build_in/build/official_unetpp.yaml)

    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd unetpp
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_unetpp.yaml
    ```

### step.4 模型推理
1. runstream推理，参考[vsx_inference.py](./build_in/vsx/python/vsx_inference.py)，修改参数并运行如下脚本
    ```bash
    python ../build_in/vsx/python/vsx_inference.py \
        --file_path  /path/to/dsb2018/dsb2018_256_val/images \
        --model_prefix_path deploy_weights/official_unetpp_run_stream_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/unetzoo-unetpp-vdsp_params.json \
        --gt_path /path/to/dsb2018/dsb2018_256_val/masks \
        --save_dir ./runstream_output \
        --device 0
    ```

### step.5 性能精度测试
1. 基于[image2npz.py](../common/utils/image2npz.py)，将评估数据集转换为npz格式（注意配置图片后缀为`.png`）：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path dsb2018/dsb2018_256_val/images \
    --target_path  dsb2018/dsb2018_256_val/images_npz \
    --text_path npz_datalist.txt
    ```

2. 性能测试，配置vdsp参数[unetzoo-unetpp-vdsp_params.json](./build_in/vdsp_params/unetzoo-unetpp-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/official_unetpp_run_stream_int8/mod \
    --vdsp_params ../build_in/vdsp_params/unetzoo-unetpp-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,96,96]
    ```

> 可选步骤，和step.4内使用runstream脚本方式的精度测试基本一致

3. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/official_unetpp_run_stream_int8/unetpp \
    --vdsp_params ../build_in/vdsp_params/unetzoo-unetpp-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,96,96] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```

4. [vamp_eval.py](./build_in/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
    --src_dir dsb2018/dsb2018_256_val/images \
    --gt_dir dsb2018/dsb2018_256_val/masks \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir deploy_weights/unetpp-int8-kl_divergence-3_96_96-debug-result \
    --input_shape 96 96 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```

### Tips

- 注意输入图像需设置为`BGR`，否则精度有损失
