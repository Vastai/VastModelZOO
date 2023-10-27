## Deeplab_v3

[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

## Code Source

```
link: https://github.com/VainF/DeepLabV3Plus-Pytorch
branch: master
commit: 4e1087de98bc49d55b9239ae92810ef7368660db
```

## Model Arch

<div  align="center">
<img src="../../images/deeplab_v3/arch.png">
</div>

### pre-processing

DeepLab_v3网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至一定尺寸(513)，然后对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(scale_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

DeepLab_v3算法的后处理即是对网络输出的heatmap进行逐像素判断，比如一共20个类别，则网络会输出21个通道(20class+background)尺寸等于原图大小的heatmap，然后逐像素判断哪个通道数值大，就表示当前像素点所代表的类别为当前通道对应的类别

### backbone

DeepLab_v3使用ResNet(50/101)作为骨架网络进行特征提取，取ResNet中最后一个block4，并在其后面增加级联模块，运用不同空洞率的空洞卷积代替传统卷积。
为捕获multi-scale context，借鉴图像金字塔、encode-decode等方法，改善v2版本中的空洞空间金字塔池化ASPP模块，加入BN层，将v2中的ASPP中尺寸3×3，dilation=24的空洞卷积替换成一个普通的1×1卷积，以保留滤波器中间部分的有效权重；增加了全局平均池化以便更好的捕捉全局信息。

<div  align="center">
<img src="../../images/deeplab_v3/aspp.png">
</div>

DeepLab_v3+的主要改进：

- 将SPP和encoder-decoder结构结合起来
- 在ASPP模块和decoder模块中引入深度可分离卷积
- Modified Aligned Xception更深的骨架网络
- 所有的最大池化结构都被stride=2的深度可分离卷积代替
- 每个3x3的深度卷积后都跟着BN和Relu
- 在Encoder部分，对压缩四次的初步有效特征层利用ASPP结构特征提取，然后进行合并，再进行1x1卷积压缩特征
- 在Decoder中，我们会对压缩两次的初步有效特征层利用1x1卷积调整通道数，与上面经过ASPP处理的特征进行连接，之后进行两次卷积操作得到最终的特征图

<div  align="center">
<img src="../../images/deeplab_v3/plus.png">
</div>

### common

- Atrous Convolution
- ASPP

## Model Info

### 模型精度

|                                Methods                                | FLOPs(G) | Params(M) |  MIoU  |            VACC MIoU            |    Shapes    |
| :--------------------------------------------------------------------: | :------: | :-------: | :----: | :-----------------------------: | :---------: |
|   [DeepLabV3-MobileNet](https://github.com/VainF/DeepLabV3Plus-Pytorch)   |  13.372  |   5.114   | 70.100 | fp16 67.816 `<br>`int8 64.441 | 3×513×513 |
|   [DeepLabV3-ResNet50](https://github.com/VainF/DeepLabV3Plus-Pytorch)   | 114.137 |  39.639  | 76.900 | fp16 72.671 `<br>`int8 72.330 | 3×513×513 |
|   [DeepLabV3-ResNet101](https://github.com/VainF/DeepLabV3Plus-Pytorch)   | 160.224 |  58.631  | 77.300 | fp16 75.289 `<br>`int8 74.492 | 3×513×513 |
| [DeepLabV3Plus-MobileNet](https://github.com/VainF/DeepLabV3Plus-Pytorch) |  37.799  |   5.226   | 71.100 |  fp16 3.589 `<br>`int8 3.570  | 3×513×513 |
| [DeepLabV3Plus-ResNet50](https://github.com/VainF/DeepLabV3Plus-Pytorch) | 139.293 |   39.76   | 77.200 | fp16 72.723 `<br>`int8 72.032 | 3×513×513 |
| [DeepLabV3Plus-ResNet101](https://github.com/VainF/DeepLabV3Plus-Pytorch) | 185.381 |  58.754  | 78.300 | fp16 75.751 `<br>`int8 75.143 | 3×513×513 |


### 测评数据集说明

[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)数据集除了用于object detection任务之外，还用于segmentation等任务，该数据集包含20个对象类，加背景共21类。

数据集子文件夹SegmentationClass内存放了JPEGimages中部分对应的pixel-level标注，以png形式存放，用于语义分割。

<div  align="center">
<img src="../../images/unet/voc.jpg" width="50%" height="50%">
</div>

### 指标说明

- IoU并交比：两个区域重叠的部分除以两个区域的集合部分，取值TP/(TP+FN+FP)
- MIoU平均并交比：分割图像一般都有好几个类别，把每个分类得出的分数进行平均得到mean IoU，也就是mIoU，其是各种基准数据集最常用的标准之一，绝大数的图像语义分割论文中模型评估比较都以此作为主要评估指标。

## VACC部署

- [vainf.md](./source_code/vainf.md)

## Tips

