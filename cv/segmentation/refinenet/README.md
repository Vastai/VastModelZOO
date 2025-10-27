## RefineNet

[RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/abs/1611.06612)

[Light-Weight RefineNet for Real-Time Semantic Segmentation](http://bmvc2018.org/contents/papers/0494.pdf)


## Code Source
```
link: https://github.com/DrSleep/refinenet-pytorch
branch: master
commit: 8f25c076016e61a835551493aae303e81cf36c53

link: https://github.com/DrSleep/light-weight-refinenet
branch: master
commit: 538fe8b39327d8343763b859daf7b9d03a05396e
```

## Model Arch
### pre-processing

RefineNet网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至一定尺寸(500)，然后对其进行归一化、减均值除方差等操作：

```python
[
    torchvision.transforms.Resize(scale_size),
    torchvision.transforms.CenterCrop(input_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

RefineNet算法的后处理即是对网络输出的heatmap进行逐像素判断，比如一共20个类别，则网络会输出21个通道(20class+background)尺寸等于原图大小的heatmap，然后逐像素判断哪个通道数值大，就表示当前像素点所代表的类别为当前通道对应的类别

### backbone

RefineNet可使用ResNet(50/101/152/mobilenetv2等)作为骨架网络进行特征提取。

语义分割方法由于采用卷积或者池化层，造成了图像分辨率的降低。为此，Lin等人提出了RefineNet，一种多路径强化网络。RefineNet的显式的利用了下采样过程的所有信息，使用远程残差连接来实现高分辨率的预测。此时，浅层的完善特征可以直接的用于强化高级的语义特征。
- 提出了多路径网络，利用多级别的抽象用于高分辨率语义分割；
- 通过使用带残差连接的同态映射构建所有组件，梯度能够在短距离和长距离传播，从而实现端到端的训练；
- 提出了链式残差池化模块，从较大的图像区域俘获背景上下文。使用多个窗口尺寸获得有效的池化特征，并使用残差连接和学习到的权重融合到一起。


<div  align="center">
<img src="../../../images/cv/segmentation/refinenet/arch.png" width="90%" height="90%">
</div>

RefineNet包括以下几种小模块：
- Residual Convolution Unit, RCU：对ResNet block进行2层的卷积操作。注意这里有多个ResNet block作为输入。
- Multi-Resolution Fusion, MRF：将1中得到的feature map进行加和融合。
- Chained Residual Pooling, CRP：该模块用于从一个大图像区域中捕捉背景上下文。注意：pooling的stride为1。
- Output convolutions：由三个RCUs构成。


<div  align="center">
<img src="../../../images/cv/segmentation/refinenet/detail.png" width="80%" height="80%">
</div>

### common

- Residual Convolution Unit
- Multi-Resolution Fusion

### 轻量化版本Light-Weight RefineNet
Light-Weight RefineNet在原始模型的基础上，作了轻量化处理：
- 使用轻量级NASNet-Mobile、MobileNet-v2骨干网络
- RCU-LW模块，在原始结构的基础上增加了1×1卷积，使得网络运算量减少
- CRP-LW模块，将原始的 3×3 卷积换为 1×1 卷积，在分割任务中，影响很小
- FUSION-LW模块，将3×3卷积换成1×1卷积

<div  align="center">
<img src="../../../images/cv/segmentation/refinenet/lw.png" width="70%" height="70%">
</div>

## Model Info

## 模型精度


| Models |  FLOPs(G) |Params(M) | MIoU | VACC int8 MIoU |VACC FP16 MIoU |
|:-:|:-:|:-:|:-:|:-:|:-:|
| [RefineNet-ResNet101](https://github.com/DrSleep/refinenet-pytorch) | 565.114 | 118.052 | 82.4 | 76.935 | 78.037 |
| [RefineNet-LW-ResNet50](https://github.com/DrSleep/light-weight-refinenet) |69.820 | 27.358 |  78.5 | 80.144 |80.270 |
| [RefineNet-LW-ResNet101](https://github.com/DrSleep/light-weight-refinenet) | 113.156 | 46.350 | 80.3 | 77.188 |77.702 |
| [RefineNet-LW-ResNet152](https://github.com/DrSleep/light-weight-refinenet) | 156.231 | 61.993 |  82.1 | 79.090 |79.602 |
| [RefineNet-LW-MobileNetv2](https://github.com/DrSleep/light-weight-refinenet) | 18.697 | 3.285 | 76.2 | 72.840 |74.483 |


> Tips：
> 
> - 原始torch模型尺寸：3x625x468，vacc尺寸：3x500x500
> 


### 测评数据集说明

[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)数据集除了用于object detection任务之外，还用于segmentation等任务，该数据集包含20个对象类，加背景共21类。

数据集子文件夹SegmentationClass内存放了JPEGimages中部分对应的pixel-level标注，以png形式存放，用于语义分割。

<div  align="center">
<img src="../../../images/cv/segmentation/unet/voc.jpg" width="50%" height="50%">
</div>


### 指标说明
- IoU并交比：两个区域重叠的部分除以两个区域的集合部分，取值TP/(TP+FN+FP)
- MIoU平均并交比：分割图像一般都有好几个类别，把每个分类得出的分数进行平均得到mean IoU，也就是mIoU，其是各种基准数据集最常用的标准之一，绝大数的图像语义分割论文中模型评估比较都以此作为主要评估指标。


## Build_In Deploy
- [drsleep.md](./source_code/drsleep.md)
