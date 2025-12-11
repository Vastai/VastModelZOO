
# NCNet

[Fast Nearest Convolution for Real-Time Image Super-Resolution](https://arxiv.org/abs/2208.11609)

## Code Source
```
# official
link: https://github.com/Algolzw/NCNet
branch: main
commit: 048486534a209c72ef6bbe991a4a926e61c18345

```

## Model Arch

<div align=center><img src="../../../images/cv//super_resolution/ncnet/ncnet.png"></div>

### pre-processing

RCAN系列网络的预处理操作，可以按照如下步骤进行（不同来源预处理和后处理可能不同，实际请参考对应推理代码）：

```python
def get_image_data(image_file, input_shape = [1, 3, 1080, 1920]):
    size = input_shape[2:][::-1]

    image = cv2.imread(image_file)
    img = cv2.resize(image, size) # , interpolation=cv2.INTER_AREA
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) # HWC to CHW
    img = np.expand_dims(img, axis=0)

    return np.array(img)
```

### post-processing

RCAN系列网络的后处理操作，可以按照如下步骤进行：
```python
heatmap = vacc_model.get_output(name, 0, 0).asnumpy().astype("float32")

output = np.squeeze(heatmap)
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
```

### backbone

文中提出一个简单的普通卷积网络与快速最近卷积模块（NCNet），它对NPU友好，并能实时执行可靠的超分辨率。所提出的最近卷积具有与最近上采样相同的性能，但速度更快。 

### common

- Pixel-Shuffle

## Model Info

### 模型性能

| Models  |  Code Source |Flops(G) | Params(M) | PSNR(dB) | SSIM | Shape |
| :---: | :--: |:--: | :--: | :---: | :----: | :--------: |
| NCNet | [Official](https://github.com/Algolzw/NCNet) |  195.748  |  0.042  |  32.917 | 0.776 | 3x1080x1920|
| NCNet **vacc max int8** |  -  |  -  |  -  |  32.448 | 0.766 |  3x1080x1920  |


> Tips
> - 此模型来自官方仓库内的torch版本
> - torch版本只提供网络文件，未提供权重；基于[yulunzhang/RCAN](https://github.com/yulunzhang/RCAN)仓库自训练
> - 精度指标基于DIV2K valid两倍放大数据集


### 测评数据集说明

[DIV2K数据集](https://data.vision.ee.ethz.ch/cvl/DIV2K/)是一个受欢迎的单图像超分辨率数据集，可用于通过低分辨率图像重建高分辨率图像。
此数据集包含 1000 张具有不同退化类型的低分辨率图像，分为：
- 训练数据：800 张低分辨率图像，并为降级因素提供高分辨率和低分辨率图像。
- 验证数据：100 张高清高分辨率图片，用于生成低分辨率的图像。
- 测试数据：100 张多样化的图像，用来生成低分辨率的图像。

<div  align="center">
<img src="../../../images/dataset/div2k.png" width="70%" height="70%">
</div>


### 评价指标说明
- 峰值信噪比(Peak Signal-to-Noise Ratio, PSNR)，PSNR是信号的最大功率和信号噪声功率之比，测量重构图像的质量，通常以分贝（dB）来表示。PSNR指标越高，说明图像质量越好
- 结构相似性评价(Structure Similarity Index, SSIM)，SSIM是衡量两幅图像相似度的指标，其取值范围为[0,1]，SSIM的值越大，表示图像失真程度越小，说明图像质量越好
- Fréchet Inception Distance，FID是衡量两个多元正态分布的距离，反映了生成图片和真实图片的距离，数据越小越好


## Build_In Deploy

### step.1 获取预训练模型
- [export.py](./source_code/official/export.py)，修改原始权重路径，导出torchscript和onnx

### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集
- 通过[image2npz.py](../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_ncnet.yaml](./build_in/build/official_ncnet.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd ncnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_ncnet.yaml
    ```

### step.4 模型推理

    - 参考[official_vsx_inference.py](./build_in/vsx/python/official_vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/official_vsx_inference.py \
        --lr_image_dir  /path/to/DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --model_prefix_path deploy_weights/official_ncnet_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-ncnet-vdsp_params.json \
        --hr_image_dir /path/to/DIV2K/DIV2K_valid_HR \
        --save_dir ./infer_output \
        --device 0
    ```

    ```
    # fp16
    mean psnr: 32.10119814424238, mean ssim: 0.8881343711575213

    # int8 
    mean psnr: 32.48110525269351, mean ssim: 0.8862011126599014
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[official-ncnet-vdsp_params.json](./build_in/vdsp_params/official-ncnet-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/official_ncnet_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-ncnet-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,1080,1920]
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，基于[image2npz.py](../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --target_path  DIV2K/DIV2K_valid_LR_bicubic/X2_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/official_ncnet_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-ncnet-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,1080,1920] \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[official-vamp_eval.py](./build_in/vdsp_params/official-vamp_eval.py)，
    ```bash
    python ../build_in/vdsp_params/official-vamp_eval.py \
        --gt_dir DIV2K/DIV2K_valid_HR \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output \
        --input_shape 1080 1920 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```


