
# DenseNet

[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

## Model Arch

<div align=center><img src="../../../images/cv/classification/densenet/arch.png"></div>

### pre-processing

DenseNet系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

DenseNet系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

DenseNet系列网络的backbone结构与ResNet类似，由连续的Dense Block和Transitin Layer组成

<div align=center><img src="../../../images/cv/classification/densenet/block.png" width="50%" height="50%"></div>

### head

DenseNet系列网络的head层由global-average-pooling层和一层全连接层组成

### common

- dense block
- Transitin Layer

## Model Info

### 模型性能

|       模型       |                                                  源码                                                  |  top1  |  top5  | MACs(G) | params(M) | input size |
| :--------------: | :----------------------------------------------------------------------------------------------------: | :----: | :----: | :-----: | :-------: | :--------: |
|   densenet121    |      [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/densenet.py)       | 74.434 | 91.972 |  2.897  |   7.979   |    224     |
|   densenet161    |      [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/densenet.py)       | 77.138 | 93.56  |  7.846  |  28.681   |    224     |
|   densenet169    |      [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/densenet.py)       |  75.6  | 92.806 |  3.436  |  14.149   |    224     |
|   densenet201    |      [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/densenet.py)       | 76.896 | 93.37  |  4.390  |  20.014   |    224     |
|   densenet121    |     [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/densenet.py)      | 75.584 | 92.652 |  2.834  |   7.895   |    224     |
|   densenet161    |     [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/densenet.py)      | 77.354 | 93.636 |  7.728  |  28.461   |    224     |
|   densenet169    |     [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/densenet.py)      | 75.898 | 93.030 |  3.360  |  13.991   |    224     |
|   densenet201    |     [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/densenet.py)      | 77.290 | 93.478 |  4.292  |  19.785   |    224     |
| densenetblur121d |     [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/densenet.py)      | 76.584 | 93.192 |  3.074  |   7.914   |    224     |
|   densenet121    | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/densenet.py) | 74.96  | 92.21  |  2.88   |   7.98    |    224     |
|   densenet161    | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/densenet.py) | 77.61  | 93.83  |  7.82   |   28.68   |    224     |
|   densenet169    | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/densenet.py) | 75.08  | 93.11  |  3.42   |   14.15   |    224     |
|   densenet201    | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/densenet.py) | 77.32  | 93.64  |  4.37   |   20.01   |    224     |
|   densenet121    |         [keras](https://github.com/keras-team/keras/blob/2.3.1/keras/applications/densenet.py)         |  75.0  |  92.3  |  2.84   |    8.1    |    224     |
|   densenet169    |         [keras](https://github.com/keras-team/keras/blob/2.3.1/keras/applications/densenet.py)         |  76.2  |  93.2  |  3.37   |   14.3    |    224     |
|   densenet201    |         [keras](https://github.com/keras-team/keras/blob/2.3.1/keras/applications/densenet.py)         |  77.3  |  93.6  |  4.30   |   20.2    |    224     |
|   densenet121    | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#DPN&DenseNet) | 75.66|  92.58  | 2.87	|8.06    |    224     |
|   densenet161    | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#DPN&DenseNet) | 78.57|  94.14  |  7.79	|28.90   |    224     |
|   densenet169    | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#DPN&DenseNet) | 76.81|  93.31  |  3.40	|14.31   |    224     |
|   densenet201    | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#DPN&DenseNet) | 77.63|  93.66	  |  4.34	|20.24   |    224     |
|   densenet264    | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#DPN&DenseNet) | 77.96|  93.85  |  5.82	| 33.74   |    224     |

### 测评数据集说明

<div align=center><img src="../../../images/dataset/imagenet.jpeg"></div>

ImageNet是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Build_In Deploy
 
 - [keras_densenet](./source_code/keras_densenet.md)
 - [mmcls_densenet](./source_code/mmcls_densenet.md)
 - [ppcls_densenet](./source_code/ppcls_densenet.md)
 - [timm_densenet](./source_code/timm_densenet.md)



