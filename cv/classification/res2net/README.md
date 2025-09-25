
# Res2Net

[Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169)



## Model Arch

### pre-processing

Res2Net网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作：

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

Res2Net系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别。

### backbone

在这项工作中，作者提出了一种简单而有效的多尺度处理方法。不像现存的方法cnn的逐层多尺度表示强度，该算法在更细粒度的层次上提高了多尺度表示能力。提出的方法的多尺度指的是更细粒度的多个可用感受野。为了实现这一目标，用一组较小的滤波器组替换n个通道的3×3滤波器，每个滤波器组有w个通道。如图所示，这些滤波器组以分层类似残差样式的连接，以增加尺度表示输出特征。具体的说，将输入特征分成了几组。一组滤波器首先从一组输入特征图中提取要素。然后将上一组的输出特征图与另一组输入特征图一起发送到下一组滤波器。此过程重复几次，直到处理完所有输入特征图。最后，将所有组的特征图连接(concat)并送到另个1×1滤波器，以完全融合信息。沿着输入特征图到输出特征图任何可能路径，当通过3×3滤波器时，等效感受野都会增加，由于组合效应，得到许多等效特征尺度。

<div align=center><img src="../../../images/cv/classification/res2net/res2net.png" width="80%" height="80%"></div>


### head

Res2Net系列网络的head层由global-average-pooling层和一层 1x1 卷积层（等同于 FC 层），GAP 后的特征便不会直接经过分类层，而是先进行了融合，并将融合的特征进行分类。

### common

- Split
- 1x1 convolution

## Model Info

### 模型性能

|       模型        |                                                       源码                                                       |  top1  |  top5  | flops(M) | params(M) | input size |
| :---------------: | :--------------------------------------------------------------------------------------------------------------: | :----: | :----: | :------: | :-------: | :--------: |
|   res2net50_14w_8s   | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/Res2Net.md) | 79.5 | 94.7 |  9.010  |   25.720   |    224     |
|   res2net50_26w_4s   | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/Res2Net.md) | 79.3 | 94.6 |  8.520  |   25.700   |    224     |
|   res2net50_vd_26w_4s   | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/Res2Net.md) | 79.8 | 94.9 |  8.370  |   25.060   |    224     |
|   res2net50_vd_26w_4s_ssld   | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/Res2Net.md) | 83.1 | 96.6 |  8.370  |   25.060   |    224     |
|   res2net101_vd_26w_4s   | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/Res2Net.md) | 80.6 | 95.2 |  16.670  |   45.220   |    224     |
|   res2net101_vd_26w_4s_ssld   | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/Res2Net.md) | 83.9 | 97.1 |  16.670  |   45.220   |    224     |
|   res2net200_vd_26w_4s   | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/Res2Net.md) | 81.2 | 95.7 |  31.490  |   76.210   |    224     |
|   res2net200_vd_26w_4s_ssld   | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/Res2Net.md) | 85.1 | 97.4 |  31.490  |   76.210   |    224     |

### 测评数据集说明

<div align=center><img src="../../../images/dataset/imagenet.jpeg"></div>

ImageNet是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Build_In Deploy

- [ppcls](./source_code/ppcls.md)