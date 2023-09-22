# MODNet


## Code Source
```
link: https://github.com/ZHKKKe/MODNet
branch: master
commit: 28165a451e4610c9d77cfdf925a94610bb2810fb
```

## Model Arch

### describe

现有的Matting方法常常需要辅助的输入如tripmap才能获得好的效果，但是tripmap获取成本较高。MODNet是一个不需要Trimap的实时抠图算法。MODNet包含2种新颖的方法来提升模型效率和鲁棒性：
- e-ASPP(Efficient Atrous Spatial Pyramid Pooling)融合多尺度特征图；
- 自监督SOC(sub-objectives consistency)策略使MODNet适应真实世界的数据。

<div  align="center">
<img src="../../images/modnet/arch.png" width="70%" height="70%">
</div>

Semantic Estimation，用来定位肖像的位置，这里仅使用了encoder来提取高级语义信息，这里的encoder可以是任意backbone网络，论文中使用mobilenetv2。这么做有2个好处：
- Semantic Estimation效率更高，因为没有decoder，参数减少了；
- 得到的高级语义表示S(I)对后续分支有利；

Efficient ASPP (e-ASPP)，DeepLab提出的ASPP已被证明可以显著提升语义分割效果，它利用多个不同空洞率的卷积来得到不同感受野的特征图，然后将多个特征图融合（ASPP可以参考这里）。为了减少计算量，对ASPP进行以下修改：
- 将每个空洞卷积改为depth-wise conv+point-wise conv；
- 交换通道融合和多尺度特征图融合的顺序，ASPP是各个通道先计算，得到不同尺度特征图然后用conv融合，e-ASPP是每个通道不同空洞率的卷积，concat后融合（这里是参考论文理解的，源码没找到这部分）；
- 输入e-ASPP的特征图通道数减少为原来的1/4。
<div  align="center">
<img src="../../images/modnet/e-aspp.png" width="70%" height="70%">
</div>


## Model Info

### 模型精度

| Model | FLOPs(G)|Params(M)  | MSE | MAD |mIOU| Shapes|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [modnet](https://github.com/ZHKKKe/MODNet) |  24.011 | 8.700| 0.00945 | 0.01456|0.970898|3x480x288|
| modnet **vacc fp16**|  - | - | 0.00951 | 0.01378| 0.970788| 3x480x288|
| modnet **vacc int8 kl**|  - | - | 0.01915 | 0.02453| 0.946498| 3x480x288|


### 测评数据集说明

[PPM-100](https://github.com/ZHKKKe/PPM)是一个人像抠图基准，它包含了100张来自Flickr的人像图片，具有以下特点：
- 精细标注 - 所有图像都被仔细标注并检查。
- 丰富多样 - 图像涵盖全身/半身人像和各种姿态。
- 高分辨率 - 图像的分辨率介于1080P和4K之间。
- 自然背景 - 所有图像都包含原始无替换的背景。

<div  align="center">
<img src="../../images/datasets/ppm-100.jpg" width="70%" height="70%">
</div>


### 指标说明
- SAD(Sum of Absolute Difference)：绝对误差和
- MAD(Mean Absolute Difference): 平均绝对差值
- MSE(Mean Squared Error)：均方误差
- MIoU平均并交比：分割图像一般都有好几个类别，把每个分类得出的分数进行平均得到mean IoU，也就是mIoU，其是各种基准数据集最常用的标准之一，绝大数的图像语义分割论文中模型评估比较都以此作为主要评估指标。


## VACC部署
- [official.md](./source_code/official.md)


## Tips
- 官方代码实现中，使用了SE_Block代替e-ASPP模块