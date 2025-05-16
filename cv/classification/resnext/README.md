
# ResNeXt

- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)


## Model Arch

<div align=center><img src="../../../images/cv/classification/resnext/arch.png" width="50%" height="50%"></div>

### pre-processing

ResNeXt系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

ResNeXt系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

ResNeXt提出了一种介于普通卷积和深度可分离卷积的这种策略：分组卷积，通过控制分组的数量（基数）来达到两种策略的平衡。分组卷积的思想是源自Inception，不同于Inception的需要人工设计每个分支，ResNeXt的每个分支的拓扑结构是相同的。最后再结合残差网络，得到的便是最终的ResNeXt。

<div align=center><img src="../../../images/cv/classification/resnext/block.png"></div>

### head

ResNeXt系列网络的head层由global-average-pooling层和一层全连接层组成

### common

- group convolution

## Model Info

### 模型性能

| 模型  | 源码 | top1 | top5 | MACs(G) | params(M) | input size |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: |
| resnext50_32x4d |[torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/resnet.py)|   77.618   |  93.698   |   4.288    |    25.029    |        224    |
| resnext101_32x8d |[torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/resnet.py)   |   79.312   |   94.526  | 16.539      |  88.791      |      224     |
| resnext50_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)   |   81.108  |   95.326  | 4.288      |  25.029      |      224      |
| resnext50d_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)   |   79.670   |   94.864  | 4.531      |  25.048      |      224      |
| resnext101_32x8d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)    | 79.316  |   94.518  | 16.539     |  88.791      |      224      |
| resnext101_64x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)  |   83.140  |   96.370  | 15.585      | 83.455       |      224      |
| tv_resnext50_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)|   77.616   |   93.700   |   4.288    |   25.029     |     224       |
| gluon_resnext50_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/gluon_resnet.py)  |   79.364  |   94.426  | 4.765      | 25.029       |      224      |
| gluon_resnext101_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/gluon_resnet.py)  |   80.344  |   94.926  | 8.95      | 44.178       |      224      |
| gluon_resnext101_64x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/gluon_resnet.py)  |   80.604  |   94.992  | 17.317      | 83.455       |      224      |
| resnext50_32x4d |[mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.2/configs/resnext/README.md)|   77.90   |   93.66   |   4.27    |   25.03     |     224       |
| resnext101_32x4d |[mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.2/configs/resnext/README.md)|   78.61   |   94.17   |   8.03    |   44.18     |     224       |
| resnext101_32x8d |[mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.2/configs/resnext/README.md)|   79.27   |   94.58   |   16.5    |   88.79     |     224       |
| resnext152_32x4d |[mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.2/configs/resnext/README.md)|   78.88   |   94.33   |   11.8    |   59.95    |     224       |


| 模型  | 源码 | top1 | top5 | FLOPS(G) | params(M) | input size |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: |
| resnext50_32x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   77.8   |   93.8   |   8.02    |   23.64    |     224       |
| resnext50_64x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   78.4   |   94.1   |   15.06    |   42.36    |     224       |
| resnext50_vd_32x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   79.6   |   94.6   |   8.5    |   23.66    |     224       |
| resnext50_vd_64x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   80.1   |   94.9   |   15.54    |   42.38    |     224       |
| resnext101_32x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   78.7   |   94.2   |   15.01    |   41.54    |     224       |
| resnext101_64x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   78.4   |   94.5   |   29.05    |   78.12    |     224       |
| resnext101_vd_32x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   80.3   |   95.1   |   15.49    |   41.56    |     224       |
| resnext101_vd_64x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   80.8   |   95.2   |   29.53    |   78.14    |     224       |
| resnext152_32x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   79.0   |   94.3   |   22.01    |  56.28   |     224       |
| resnext152_64x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   79.5   |   94.7   |   43.03    |   107.57    |     224       |
| resnext152_vd_32x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   80.7   |   95.2   |   22.49    |  56.3   |     224       |
| resnext152_vd_64x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   81.1   |   95.3   |   43.52    |   107.59    |     224       |
| resnext101_32x8d_wsl |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/EfficientNet_and_ResNeXt101_wsl.md)|   82.6   |   96.7   |   29.14    |   78.44    |     224       |
| resnext101_32x16d_wsl |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/EfficientNet_and_ResNeXt101_wsl.md)|   84.2   |   97.3   |  57.55    |   152.66    |     224       |
| resnext101_32x32d_wsl |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/EfficientNet_and_ResNeXt101_wsl.md)|  85.0   |   97.6   |   115.17    |  303.11    |     224       |

**Note:** `fix_resnext101_32x48d_wsl`模型与`resnext101_32x48d_wsl`模型在转换onnx格式时失败

### 测评数据集说明

<div align=center><img src="../../../images/dataset/imagenet.jpeg"></div>

ImageNet是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Build_In Deploy

- [mmcls_resnext.md](source_code/mmcls_resnext.md)
- [ppcls_resnext.md](source_code/ppcls_resnext.md)
- [timm_resnext.md](source_code/timm_resnext.md)
- [torchvision_resnext.md](source_code/torchvision_resnext.md)