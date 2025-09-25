
# Mobileone

[MobileOne: An Improved One millisecond Mobile Backbone](https://arxiv.org/abs/2206.04040)

## Code Source
```
link: https://github.com/apple/ml-mobileone.git
commit: b7f4e6d4888

link: https://github.com/open-mmlab/mmpretrain.git
tag: v1.0.0rc7

```


## Model Arch

<div align=center><img src="../../../images/cv/classification/mobileone/mobileone.png"></div>

### pre-processing


Mobileone系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],),
]
```

### post-processing

Mobileone系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

Mobileone Block 基于深度可分离卷积，由多分支的 DepthWise 卷积模块和 PointWise 卷积模块组成，它有两种状态，训练时状态和推理时状态。在训练状态下，DepthWise 卷积模块有三个分支，分别为 1×1 DepthWise 卷积分支，3×3 DepthWise 卷积分支和一个 BN 层分支。PointWise 卷积模块有两个分支，分别为 1×1 卷积分支和一个 BN层 分支。
在推理状态下， DepthWise 卷积模块 和 PointWise 卷积模块只有一个 3×3 DepthWise 卷积和 1×1 PointWise 卷积，没有其他额外的分支，其中卷积和 BN 还可以进一步融合。仅mobileone-s4使用少量SE模块

### head

Mobileone系列网络的head层由global-average-pooling层和一层全连接层组成

### common

- DepthWise conv
- PointWise conv
- Squeeze and Excite module

## Model Info

### 模型性能

| 模型  | 源码 | top1 | top5 | flops(G) | params(M) | input size |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: |
| mobileone_s0 |[official](https://github.com/apple/ml-mobileone)|   71.4   |      |   0.611    |    2.079    |        224    |
| mobileone_s1 |[official](https://github.com/apple/ml-mobileone)   |   75.9   |     | 1.834      |  4.765      |      224     |
| mobileone_s2 |[official](https://github.com/apple/ml-mobileone)   |   77.4   |    | 2.886      |  7.808      |      224      |
| mobileone_s3 |[official](https://github.com/apple/ml-mobileone)   |   78.1   |     | 4.213     |  10.078      |      224      |
| mobileone_s4 |[official](https://github.com/apple/ml-mobileone)    | 79.4   |     | 6.621      |  14.838      |      224      |


### 测评数据集说明

<div align=center><img src="../../../images/dataset/imagenet.jpeg"></div>

[ImageNet](https://image-net.org/challenges/LSVRC/2012/index.php)是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Build_In Deploy

- [official_mobileone](./source_code/official_mobileone.md)