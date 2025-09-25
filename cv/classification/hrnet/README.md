
# HRNet

[Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/pdf/1908.07919.pdf)


## Model Arch

<div align=center><img src="../../../images/cv/classification/hrnet/cls-hrnet.png"></div>

### pre-processing

HRNet系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

HRNet系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

HRNet网络是将不同分辨率的feature map进行并联，在并联的基础上添加不同分辨率feature map之间的融合，具体融合的方法可以分为4种：

1. 同分辨率的层直接复制
2. 需要升分辨率的使用bilinear upsample + 1x1卷积将channel数统一
3. 需要降分辨率的使用stride为2的3x3 卷积
4. 三个feature map融合的方式是相加

通过上述规则生成了一系列特征层的组合，然后选择相应的特征组合，即组成了基于HRNet算法的分类网络

### head

HRNet系列网络的head层由global-average-pooling层和一层全连接层组成

### common

- bilinear upsample

## Model Info

### 模型性能

|           模型           |                                                源码                                                 |  top1  |  top5  | flops(G) | params(M) | input size |
| :----------------------: | :-------------------------------------------------------------------------------------------------: | :----: | :----: | :------: | :-------: | :--------: |
| HRNet_w18_small_model_v1 |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  72.3  |  90.7  |   1.49   |   13.2    |    224     |
| HRNet_w18_small_model_v2 |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  75.1  |  92.4  |   2.42   |   15.6    |    224     |
|        HRNet_w18         |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  76.8  |  93.3  |   3.99   |   21.3    |    224     |
|        HRNet_w30         |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  78.2  |  94.2  |   7.55   |   37.7    |    224     |
|        HRNet_w32         |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  78.5  |  94.2  |   8.31   |   41.2    |    224     |
|        HRNet_w40         |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  78.9  |  94.5  |   11.8   |   57.6    |    224     |
|        HRNet_w44         |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  78.9  |  94.4  |   13.9   |   67.1    |    224     |
|        HRNet_w48         |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  79.3  |  94.5  |   16.1   |   77.5    |    224     |
|        HRNet_w64         |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  79.5  |  94.6  |   26.9   |   128.1   |    224     |
|        HRNet_w18         | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py) | 76.75  | 93.44  |   4.33   |   21.30   |    224     |
|        HRNet_w30         | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py) | 78.19  | 94.22  |   8.17   |   37.71   |    224     |
|        HRNet_w32         | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py) | 78.44  | 94.19  |   8.99   |   41.23   |    224     |
|        HRNet_w40         | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py) | 78.94  | 94.47  |  12.77   |   57.55   |    224     |
|        HRNet_w44         | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py) | 78.88  | 94.37  |  14.96   |   67.06   |    224     |
|        HRNet_w48         | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py) | 79.32  | 94.52  |  17.36   |   77.47   |    224     |
|        HRNet_w64         | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py) | 79.46  | 94.65  |  29.00   |  128.06   |    224     |
|    hrnet_w18_small_v1    |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 72.336 | 90.68  |  3.611   |  13.187   |    224     |
|    hrnet_w18_small_v2    |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 75.11  | 92.416 |  5.856   |  15.597   |    224     |
|        hrnet_w18         |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 76.76  | 93.444 |  9.667   |  21.299   |    224     |
|        hrnet_w30         |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 78.198 | 94.224 |  18.207  |  37.712   |    224     |
|        hrnet_w32         |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 78.452 | 94.188 |  20.026  |  41.233   |    224     |
|        hrnet_w40         |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 78.922 | 94.47  |  28.436  |  57.557   |    224     |
|        hrnet_w44         |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 78.896 | 94.37  |  33.320  |  67.065   |    224     |
|        hrnet_w48         |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      |  79.3  | 94.514 |  38.658  |  77.470   |    224     |
|        hrnet_w64         |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 79.47  | 94.654 |  64.535  |  128.060  |    224     |
|        hrnet_w18_c         |     [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#hrnet-系列-13)      | 76.92|  93.39	 |  4.32| 21.35   |    224     |
|        hrnet_w18_c_ssld    |     [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#hrnet-系列-13)      | 81.162|  95.804	 |  4.32| 21.35   |    224     |
|        hrnet_w30_c    |     [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#hrnet-系列-13)      | 78.04| 94.02	 |  8.15| 37.78  |    224     |
|        hrnet_w32_c    |     [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#hrnet-系列-13)      | 78.28	 |94.24	 | 8.97	 |41.30  |    224     |
|        hrnet_w40_c    |     [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#hrnet-系列-13)      | 78.77	 |94.47	 | 12.74	 |57.64  |    224     |
|        hrnet_w44_c    |     [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#hrnet-系列-13)      | 79.00	 |94.51		 |14.94	 |67.16 |    224     |
|        hrnet_w48_c	|     [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#hrnet-系列-13)      | 78.95	 |94.42	| 17.34	 |77.57 |    224     |
|        hrnet_w48_c_ssld   |     [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#hrnet-系列-13)      | 83.63	 |96.82	| 17.34	 |77.57|    224     |
|        hrnet_w64_c	    |     [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#hrnet-系列-13)      | 79.30	 |94.61	| 28.97	 |128.18|    224     |
|        se_hrnet_w64_c_ssld|     [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#hrnet-系列-13)      | 84.75	 |97.26	| 29.00	 |129.12 |    224     |


### 测评数据集说明

<div align=center><img src="../../../images/dataset/imagenet.jpeg"></div>

ImageNet是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Build_In Deploy

- [mmcls_hrnet.md](./source_code/mmcls_hrnet.md)
- [official_hrnet.md](./source_code/official_hrnet.md)
- [ppcls_hrnet.md](./source_code/ppcls_hrnet.md)
- [timm_hrnet.md](./source_code/timm_hrnet.md)
