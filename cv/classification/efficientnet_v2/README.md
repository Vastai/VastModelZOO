# EfficientNet_V2

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)

## Model Arch

### pre-processing

EfficientNetV2系列网络的预处理操作和常见分类算法一致，主要区别是每个子模型的输入大小均不一样

```python
[
    torchvision.transforms.Resize(),
    torchvision.transforms.CenterCrop(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
]
```

### post-processing

EfficientNetV2系列网络的后处理操作是对网络输出进行`softmax`作为每个类别的预测值，然后根据预测值进行排序，选择`topk`作为输入图片的预测分数以及类别

### backbone

EfficientNetV2中除了使用到MBConv模块外，还使用了Fused-MBConv模块（主要是在网络浅层中使用，将浅层MBConv结构替换成Fused-MBConv结构能够明显提升训练速度），Fused-MBConv结构非常简单，即将原来的MBConv结构（之前在将EfficientNetv1时有详细讲过）主分支中的expansion conv1x1和depthwise conv3x3替换成一个普通的conv3x3，如下图所示

<div align=center><img src="../../../images/cv/classification//efficientnetv2/fused-mbconv.png"></div>

论文基于trainning-aware NAS framework，搜索工作主要还是基于之前的Mnasnet以及EfficientNet. 但是这次的优化目标联合了accuracy、parameter efficiency以及trainning efficiency三个维度。下表为efficientnetv2_s的模型架构

<div align=center><img src="../../../images/cv/classification/efficientnetv2/arch.png"></div>

另外，论文提出了改进的渐进学习方法，该方法会根据训练图像的尺寸动态调节正则方法(例如dropout、data augmentation和mixup)。通过实验展示了该方法不仅能够提升训练速度，同时还能提升准确率。

### head

EfficientNetV2系列网络的head层由`AdaptiveAvgPool2d`层和`Linear`组成

### common

- SE
- depthwise
- MBConv
- Fused-MBConv


## Model Info

### 模型性能

|        模型         |                                               源码                                                |  top1  |  top5  | FLOPs (G) | Params(M) | input size |
| :-----------------: | :-----------------------------------------------------------------------------------------------: | :----: | :----: | :-------: | :-------: | :--------: |
| efficientnetv2_rw_t | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 82.344 | 96.196 |   7.032   |  13.556   |    288     |
| efficientnetv2_rw_s | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 83.810 | 96.724 |  19.201   |  23.780   |    384     |
| efficientnetv2_rw_m | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 84.812 | 97.146 |  47.406   |  52.929   |    416     |



### 测评数据集说明

<div align=center><img src="../../../images/dataset/imagenet.jpeg"></div>

ImageNet是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。`1K`代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Build_In Deploy

### step.1 获取模型

```bash
pip install timm==0.6.5
python ../common/utils/export_timm_torchvision_model.py --model_library timm  --model_name efficientnetv2_rw_t --save_dir ./onnx  --size 224 --pretrained_weights xxx.pth
```

### step.2 获取数据集
- [校准数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [评估数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [label_list](../../common/label//imagenet.txt)
- [label_dict](../../common/label//imagenet1000_clsid_to_human.txt)

### step.3 模型转换

1. 参考瀚博训推软件生态链文档，获取模型转换工具: [vamc v3.0+](../../../../docs/vastai_software.md)

2. 根据具体模型，修改编译配置
    - [timm_efficientnet.yaml](../build_in/build/timm_efficientnet.yaml)
    
    > - runmodel推理，编译参数`backend.type: tvm_runmodel`
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

3. 模型编译

    ```bash
    cd efficientnet_v2
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/timm_efficientnet.yaml
    ```

### step.4 模型推理

1. 参考瀚博训推软件生态链文档，获取模型推理工具：[vaststreamx v2.8+](../../../../docs/vastai_software.md)

2. runstream
    - 参考：[classification.py](../../common/vsx/classification.py)
    ```bash
    python ../../common/vsx/classification.py \
        --infer_mode sync \
        --file_path path/to/ILSVRC2012_img_val \
        --model_prefix_path deploy_weights/timm_efficientnetv2_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/timm-efficientnetv2_rw_t-vdsp_params.json \
        --label_txt path/to/imagenet.txt \
        --save_dir ./runstream_output \
        --save_result_txt result.txt \
        --device 0
    ```

    - 精度评估
    ```
    python ../../common/eval/eval_topk.py ./runmstream_output/mod.txt
    ```

    ```
    # fp16
    top1_rate: 81.916 top5_rate: 95.93

    # int8
    top1_rate: 77.07 top5_rate: 94.372
    ```

### step.5 性能测试
1. 参考瀚博训推软件生态链文档，获取模型性能测试工具：[vamp v2.4+](../../../../docs/vastai_software.md)

2. 性能测试
    - 配置[timm-efficientnetv2_rw_t-vdsp_params.json](../build_in/vdsp_params/timm-efficientnetv2_rw_t-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/timm_efficientnetv2_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/timm-efficientnetv2_rw_t-vdsp_params.json  -i 8 -p 1 -b 2 -s [3,224,224]
    ```

3. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致
    
    - 数据准备，生成推理数据`npz`以及对应的`dataset.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path ILSVRC2012_img_val --target_path  input_npz  --text_path imagenet_npz.txt
    ```

    - vamp推理获取npz文件
    ```
    vamp -m deploy_weights/timm_efficientnetv2_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/timm-efficientnetv2_rw_t-vdsp_params.json  -i 8 -p 1 -b 22 -s [3,224,224] --datalist imagenet_npz.txt --path_output output
    ```

    - 解析输出结果用于精度评估，参考：[vamp_npz_decode.py](../../common/eval/vamp_npz_decode.py)
    ```bash
    python  ../../common/eval/vamp_npz_decode.py imagenet_npz.txt output imagenet_result.txt imagenet.txt
    ```
    
    - 精度评估，参考：[eval_topk.py](../../common/eval/eval_topk.py)
    ```bash
    python ../../common/eval/eval_topk.py imagenet_result.txt
    ```

## appending
1. EfficientNetV2 系列模型使用int8`PQT`方案基本都会掉点
2. int8 `PQT` 效果较好的是`percentile`, gap在5个点左右
