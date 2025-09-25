
# ShuffleNetV1

[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.html)

## Model Arch

<div align=center><img src="../../../images/cv/classification/shufflenetv1/block.png" width="50%" height="50%"></div>

### pre-processing

ShuffleNetV1系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

ShuffleNetV1系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

shufflenet v1主要提出了pointwise group convolution 和 channel shuffle 结构，在保持模型精度的前提下，进一步减小了网络的计算量

### head

ShuffleNetV1系列网络的head层由global-average-pooling层和一层全连接层组成

### common

- pointwise group convolution
- channel shuffle

## Model Info

### 模型性能

| 模型  | 源码 | top1 | top5 | MACs(G) | params(M) | input size |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: |
| shufflenet_v1_x1.0 |[mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/shufflenet_v1.py)|  68.13   |   87.81   |   0.146    |    1.87    |        224    |
| ShuffleNetV1 0.5x (group=3) |[megvii](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1)|  57.3   |   80   |   0.038    |    0.7    |        224    |
| ShuffleNetV1 0.5x (group=8) |[megvii](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1)|  58.8   |   81   |   0.04    |    1.0    |        224    |
| ShuffleNetV1 1.0x (group=3) |[megvii](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1)|  67.8   |   87.7   |   0.138    |    1.9    |        224    |
| ShuffleNetV1 1.0x (group=8) |[megvii](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1)|  68   |   86.4   |   0.138   |    2.4    |        224    |
| ShuffleNetV1 1.5x (group=3) |[megvii](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1)|  71.6   |   90.2   |   0.292    |    3.4    |        224    |
| ShuffleNetV1 1.5x (group=8) |[megvii](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1)|  71   |   89.6   |   0.290    |    4.3    |        224    |
| ShuffleNetV1 2.0x (group=3) |[megvii](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1)|  74.1   |   91.4   |   0.524    |    5.4    |        224    |
| ShuffleNetV1 2.0x (group=8) |[megvii](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1)|  72.9   |   90.8   |   0.522    |    6.5    |        224    |

### 测评数据集说明

<div align=center><img src="../../../images/dataset/imagenet.jpeg"></div>

ImageNet是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Build_In Deploy

- [megvii_shufflenet_v1.md](./source_code/megvii_shufflenet_v1.md)
- [mmcls_shufflenet_v1.md](./source_code/mmcls_shufflenet_v1.md)