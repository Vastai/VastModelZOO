
![](../../../images/cv/detection/nanodet/Title.jpg)

# NanoDet

- [offical code](https://github.com/RangiLyu/nanodet)


## Model Arch

![](../../../images/cv/detection/nanodet/nanodet-plus-arch.png)

### pre-processing

`nanodet`系列算法的预处理主要是对输入图片利用`letterbox`算子进行resize，然后进行归一化后减均值除方差操作后送入网络forward即可，均值方差设置如下:

```python
mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
```

### post-processing

`nanodet`系列的后处理操作是利用网络预测特征图进行box decode，然后进行nms操作

### backbone

在NanoDet系列算法中，使用shufflenet v2作为backbone，同时作者重新设计了一个非常轻量但性能不错的PAN：Ghost-PAN。Ghost-PAN使用GhostNet中的GhostBlock作为处理多层之间特征融合的模块，其基本结构单元由一组1x1卷积和3x3的depthwise卷积组成，参数量和计算量都非常小，具体结构如下图

![](../../../images/cv/detection/nanodet/ghost-pan.webp)

因此最终整个Ghost-PAN的参数量只有190k个参数，且在ARM上只增加了大概1ms的延时，x86端和GPU端的速度影响就更小了，但是小归小，它的性能一点也不差，在增加了GhostPAN后，模型的mAP提升了2个点！


### head

ThunderNet的文章中提出，在轻量级模型中将深度可分离卷积的depthwise部分从3x3改成5x5，能够在增加较少的参数量的情况下提升检测器的感受野并提升性能。现在，在轻量级模型的depthwise部分增大kernel已经成为了非常通用的技巧，因此NanoDet-Plus也将检测头的depthwise卷积的卷积核大小也改成了5x5。

PicoDet在原本NanoDet的3层特征基础上增加了一层下采样特征，为了能够赶上其性能，NanoDet-Plus中也采取了这种改进。这部分操作增加了大约0.7mAP。

### common

- letterbox
- Ghost-PAN
- depthwise conv


## Model Info

### 模型性能

| 模型  | 源码 | mAP@.5:.95  | flops(G) | params(M) | input size |
| :---: | :--: | :--: | :--: | :----: | :--------: |
| nanodet_m |[official](https://github.com/RangiLyu/nanodet)|   20.6      | 0.72 |    0.95    |    320    |
| nanodet_plus_m |[official](https://github.com/RangiLyu/nanodet)|  27.0    | 0.9 | 1.17 | 320 |
| nanodet_plus_m |[official](https://github.com/RangiLyu/nanodet)|  30.4     | 1.52 | 1.17 | 416 |
| nanodet_plus_m_1.5x |[official](https://github.com/RangiLyu/nanodet)|  29.9    | 1.75 | 2.44 | 320 |
| nanodet_plus_m_1.5x |[official](https://github.com/RangiLyu/nanodet)|  34.1    | 2.97 | 2.44 | 416 |

### 测评数据集说明

![](../../../images/dataset/coco.png)

[MS COCO](https://cocodataset.org/#download)的全称是Microsoft Common Objects in Context，是微软于2014年出资标注的Microsoft COCO数据集，与ImageNet竞赛一样，被视为是计算机视觉领域最受关注和最权威的比赛数据集之一。

COCO数据集支持目标检测、关键点检测、实例分割、全景分割与图像字幕任务。在图像检测任务中，COCO数据集提供了80个类别，验证集包含5000张图片，上表的结果即在该验证集下测试。

### 评价指标说明

- mAP: mean of Average Precision, 检测任务评价指标，多类别的AP的平均值；AP即平均精度，是Precision-Recall曲线下的面积
- mAP@.5: 即将IoU设为0.5时，计算每一类的所有图片的AP，然后所有类别求平均，即mAP
- mAP@.5:.95: 表示在不同IoU阈值（从0.5到0.95，步长0.05）上的平均mAP

## Build_In Deploy

- [mmdet](./source_code/nanodet_deploy.md)

