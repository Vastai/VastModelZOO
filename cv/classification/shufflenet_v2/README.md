
# ShuffleNetV2

[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)


## Model Arch

<div align=center><img src="../../../images/cv/classification/shufflenetv2/arch.png" width="50%" height="50%"></div>

### pre-processing

ShuffleNetV2系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

ShuffleNetV2系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

ShuffleNetV2系列网络利用channel split将输入通道切分为两个分支，其中一个分支保持不变，另一个分支包含三个恒等通道数的卷积层，之后再将两个分组concat为一个，再执行通道随机化操作

<div align=center><img src="../../../images/cv/classification/shufflenetv2/s-block.png" width="50%" height="50%"></div>

### head

ShuffleNetV2系列网络的head层由global-average-pooling层和一层全连接层组成

### common

- channel split
- channel shuffle
- depthwise conv
- group conv

## Model Info

### 模型性能

| 模型  | 源码 | top1 | top5 | flops(M) | params(M) | input size |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: |
| shufflenet_v2_x0.5 |[torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/shufflenetv2.py)|  60.552   |   81.746   |   44.572    |    1.367    |        224    |
| shufflenet_v2_x1.0 |[torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/shufflenetv2.py)|  69.362   |   88.316   |   152.709    |    2.279    |        224    |
| shufflenet_v2_x1.0 |[mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.1/mmcls/models/backbones/shufflenet_v2.py)|  69.550   |   88.920   |   149.000    |    2.280    |        224    |
| shufflenet_v2_x0.25 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)|  49.900   |   73.790   |   18.950    |    0.610    |        224    |
| shufflenet_v2_x0.33 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)|  53.730   |   77.050   |   24.040    |    0.650    |        224    |
| shufflenet_v2_x0.5 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)|  60.320   |   82.260   |   42.580    |    1.370    |        224    |
| shufflenet_v2_x1.0 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)|  68.800   |   88.450   |   148.860    |    2.290    |        224    |
| shufflenet_v2_x1.5 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)|  71.630   |   90.150   |   301.35    |    3.530    |        224    |
| shufflenet_v2_x2.0 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)|  73.150   |   91.200   |   571.700    |    7.400    |        224    |
| shufflenet_v2_swish |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)|  70.030   |   89.170   |   148.860    |    2.290    |        224    |
| shufflenet_v2_x0.5 |[megvii](https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2/README.md)| 61.1   |   82.6   |   41    |    1.4    |        224    |
| shufflenet_v2_x1.0 |[megvii](https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2/README.md)|  69.4   |   88.9   |   146    |    2.3    |        224    |
| shufflenet_v2_x1.5 |[megvii](https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2/README.md)|  72.6   |   90.6  |   299    |    3.5    |        224    |
| shufflenet_v2_x2.0 |[megvii](https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2/README.md)|  75   |   92.4   |   591    |    7.4    |        224    |

### 测评数据集说明

<div align=center><img src="../../../images/dataset/imagenet.jpeg"></div>

ImageNet是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Build_In Deploy

- [megvii_shufflenetv2.md](./source_code/megvii_shufflenetv2.md)
- [mmcls_shufflenetv2.md](./source_code/mmcls_shufflenetv2.md)
- [ppcls_shufflenetv2.md](./source_code/ppcls_shufflenetv2.md)
- [torchvision_shufflenetv2.md](./source_code/torchvision_shufflenetv2.md)