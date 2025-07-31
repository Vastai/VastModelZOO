
# MobileNetV2

[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

## Model Arch

<div align=center><img src="../../../images/cv/classification/mobilenetv2/mobilenet_v2_structure.png" width="50%" height="50%"></div>

### pre-processing

MobileNetV2网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

MobileNetV2网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

MobileNetV2网络的backbone结构基于倒置残差结构(inverted residual structure)，原本的残差结构的主分支是有三个卷积，两个逐点卷积通道数较多，而倒置的残差结构刚好相反，中间的卷积通道数(依旧使用深度分离卷积结构)较多，旁边的较小。此外，MobileNetV2网络去除主分支中的非线性变换层，可以保持模型表现力

下图为倒置残差结构(inverted residual structure)示意图

<div align=center><img src="../../../images/cv/classification/mobilenetv2/inverted_residual.png" width="50%" height="50%"></div>

综上，mobilenetv2对应block结构如下图
<div align=center><img src="../../../images/cv/classification/mobilenetv2/arch.png" width="50%" height="50%"></div>


### head

MobileNetV2网络的head层由global-average-pooling层和一层全连接层组成

### common

- inverted residual structure
- depthwise conv


## Model Info

### 模型性能


|       模型        |                                                                  源码                                                                   |  top1  |  top5  | flops(M) | params(M) | input size |
| :---------------: | :-------------------------------------------------------------------------------------------------------------------------------------: | :----: | :----: | :------: | :-------: | :--------: |
|    mobilenetv2    |                     [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/mobilenetv2.py)                      | 71.878 | 90.286 | 327.650  |   3.505   |    224     |
|    mobilenetv2    | [timm](https://github.com/rwightman/pytorch-image-models/blob/3599c7e6a4b781cb6147f0cbdceb2e455c36fe03/timm/models/efficientnet.py#L92) | 72.956 | 91.010 | 327.544  |   3.500   |    224     |
| MobileNetV2_x0_25 |              [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/mobilenet_v2.py)              | 53.210 | 76.520 |  34.180  |   1.53    |    224     |
| MobileNetV2_x0_5  |              [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/mobilenet_v2.py)              | 65.030 | 85.720 |  99.480  |   1.98    |    224     |
| MobileNetV2_x0_75 |              [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/mobilenet_v2.py)              | 69.830 | 89.010 | 197.370  |   2.65    |    224     |
|    MobileNetV2    |              [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/mobilenet_v2.py)              | 72.150 | 90.650 | 327.840  |   3.540   |    224     |
| MobileNetV2_x1_5  |              [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/mobilenet_v2.py)              | 74.120 | 91.670 | 702.350  |   6.900   |    224     |
| MobileNetV2_x2_0  |              [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/mobilenet_v2.py)              | 75.230 | 92.580 | 1217.250 |  11.330   |    224     |
| MobileNetV2_ssld  |              [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/mobilenet_v2.py)              | 76.740 | 93.390 |  327.84  |   3.540   |    224     |



### 测评数据集说明

<div align=center><img src="../../../images/dataset/imagenet.jpeg"></div>

ImageNet是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Build_In Deploy

- [ppcls_mobilenetv2.md](source_code/ppcls_mobilenetv2.md)
- [timm_mobilenetv2.md](source_code/timm_mobilenetv2.md)
- [torchvision_mobilenetv2.md](source_code/torchvision_mobilenetv2.md)