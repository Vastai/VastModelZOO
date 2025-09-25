
# inception_v3

[inception_v3: Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)


## Model Arch

### pre-processing

inception_v3系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至342的尺寸，然后利用`CenterCrop`算子crop出299的图片对其进行归一化、减均值除方差等操作。需要注意的是，inception_v3系列所用到的均值方差与其他resnet、vgg等网络所用的均值方差有所不同

```python
[
    torchvision.transforms.Resize(342),
    torchvision.transforms.CenterCrop(299),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],),
]
```

### post-processing

inception_v3系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

inception_v3在之前的基础上增加：
- 标签平滑
- 将大卷积分解成小卷积，使得在感受野不变的情况下，减少参数的计算量
- max pooling层在下采样会导致信息损失大，于是设计成计算输入A的卷积结果，计算输入A的pooling结果，并且将卷积的结果与池化的结果concat。这样减少计算量又减少信息损失。

<div align=center><img src="../../../images/cv/classification/inceptionv3/inceptionv3.png" width="50%" height="50%"></div>

### head

inception_v3系列网络的head层由global-average-pooling层和一层全连接层组成

### common

- inception_v3架构

## Model Info

### 模型性能

|        模型        |                                               源码                                                |  top1  |  top5  | flops(G) | params(M) | input size |
| :----------------: | :-----------------------------------------------------------------------------------------------: | :----: | :----: | :------: | :-------: | :--------: |
|    inception_v3    |   [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/inception.py)    | 77.294 | 93.450 |  11.021  |  27.200   |    299     |
|    inception_v3    | [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/inception_v3.py) | 77.438 | 93.476 |  11.498  |  23.830   |    299     |
|  tf_inception_v3   | [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/inception_v3.py) | 77.852 | 93.640 |  11.498  |  23.830   |    299     |
|  adv_inception_v3  | [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/inception_v3.py) | 77.578 | 93.738 |  11.498  |  23.830   |    299     |
| gluon_inception_v3 | [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/inception_v3.py) | 78.806 | 94.370 |  11.498  |  23.830   |    299     |
|    inception_v3    |  [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/Inception.md)   | 79.100 | 94.600 |  11.460  |  23.830   |    299     |

### 测评数据集说明

<div align=center><img src="../../../images/dataset/imagenet.jpeg"></div>

ImageNet是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Build_In Deploy

- [ppcls_inceptionv3.md](./source_code/ppcls_inceptionv3.md)
- [timm_inceptionv3.md](./source_code/timm_inceptionv3.md)
- [torchvision_inceptionv3.md](./source_code/torchvision_inceptionv3.md)