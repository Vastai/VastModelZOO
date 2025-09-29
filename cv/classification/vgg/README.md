
# VGG

[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)


## Model Arch

<div align=center><img src="../../../images/cv/classification/vgg/arch.png" width="50%" height="50%"></div>

### pre-processing

VGG系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

VGG系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

VGG系列网络的backbone结构可以看成是数个vgg_block的堆叠，每个vgg_block由多个conv+bn+relu或conv+relu，最好再加上池化层组成。VGG网络名称后面的数字表示整个网络中包含参数层的数量（卷积层或全连接层，不含池化层）

### head

VGG系列网络的head层为3个全连接层组成

### common

- maxpool

## Model Info

### 模型性能

|   模型   |                                                 源码                                                  |  top1  |  top5  | flops(G) | params(M) | input size |
| :------: | :---------------------------------------------------------------------------------------------------: | :----: | :----: | :------: | :-------: | :--------: |
|  vgg11   |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 69.028 | 88.626 |  7.609   |  132.863  |    224     |
| vgg11_bn |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 70.360 | 89.802 |  7.639   |  132.869  |    224     |
|  vgg13   |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 69.926 | 89.246 |  11.308  |  133.048  |    224     |
| vgg13_bn |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 71.594 | 90.376 |  11.357  |  133.054  |    224     |
|  vgg16   |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 71.590 | 90.382 |  15.47   |  138.358  |    224     |
| vgg16_bn |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 73.350 | 91.504 |  15.524  |  138.366  |    224     |
|  vgg19   |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 72.366 | 90.870 |  19.632  |  143.667  |    224     |
| vgg19_bn |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 74.214 | 91.848 |  19.691  |  143.678  |    224     |
|  vgg11   |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 69.02  | 88.626 |  7.609   |  132.863  |    224     |
| vgg11_bn |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 70.37  | 89.81  |  7.639   |  132.869  |    224     |
|  vgg13   |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 69.928 | 89.246 |  11.308  |  133.048  |    224     |
| vgg13_bn |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 71.586 | 90.374 |  11.357  |  133.054  |    224     |
|  vgg16   |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 71.592 | 90.382 |  15.47   |  138.358  |    224     |
| vgg16_bn |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 73.36  | 91.516 |  15.524  |  138.366  |    224     |
|  vgg19   |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 72.376 | 90.876 |  19.632  |  143.667  |    224     |
| vgg19_bn |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 74.218 | 91.842 |  19.691  |  143.678  |    224     |
|  vgg11   |  [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg11_8xb32_in1k.py)  | 68.75  | 88.87  |   7.63   |  132.86   |    224     |
| vgg11_bn | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg11bn_8xb32_in1k.py) | 70.75  | 90.12  |   7.64   |  132.87   |    224     |
|  vgg13   |  [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg13_8xb32_in1k.py)  | 70.02  | 89.46  |  11.34   |  133.05   |    224     |
| vgg13_bn | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg13bn_8xb32_in1k.py) | 72.15  | 90.71  |  11.36   |  133.05   |    224     |
|  vgg16   |  [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg16_8xb32_in1k.py)  | 71.62  | 90.49  |   15.5   |  138.36   |    224     |
| vgg16_bn |  [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg16_8xb32_in1k.py)  | 73.72  | 91.68  |  15.53   |  138.37   |    224     |
|  vgg19   |  [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg19_8xb32_in1k.py)  | 72.41  | 90.80  |  19.67   |  143.67   |    224     |
| vgg19_bn | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg19bn_8xb32_in1k.py) | 74.70  | 92.24  |   19.7   |  143.68   |    224     |
|  vgg11   |  [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/VGG.md)  | 69.3  | 89.1  |   15.09   |  132.85   |    224     |
|  vgg13   |  [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/VGG.md)  | 70.0  | 89.4  |  22.48   |  133.03   |    224     |
|  vgg16   |  [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/VGG.md)  | 72.0  | 90.7  |   30.81   |  138.34   |    224     |
|  vgg19   |  [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/VGG.md)  | 72.6  | 90.9  |  39.13   |  143.65   |    224     |

### 测评数据集说明

<div align=center><img src="../../../images/dataset/imagenet.jpeg"></div>

ImageNet是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数


## Build_In Deploy

- [mmpretrain_vgg.md](./source_code/mmpretrain_vgg.md)
- [ppcls_vgg.md](./source_code/ppcls_vgg.md)
- [timm_vgg.md](./source_code/timm_vgg.md)
- [torchvision_vgg.md](./source_code/torchvision_vgg.md)

