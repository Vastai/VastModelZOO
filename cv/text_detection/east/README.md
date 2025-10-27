# EAST
[EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155)

## code source

```
# ppocr
link: https://github.com/PaddlePaddle/PaddleOCR
branch: v2.7
commit: b17c2f3a5687186caca590a343556355faacb243
```

## Model Arch

EAST文本检测算法是cvpr2017提出的，可以检测任意四边形形状的文本。

EAST做文本检测只需要两步：先是一个全卷积的网络直接产生一个字符或者文本行的预测（可以是旋转的矩形或者不规则四边形），然后通过NMS（Non-Maximum Suppression）算法合并最后的结果。

EAST网络是一个全卷积网络，主要有三部分：特征提取层，特征融合层，输出层。由于在一张图片中，各个文字大小不一，所以需要融合不同层次的特征图，小文字的预测需要用到底层的语义信息，大文字的预测要用到高层的语义信息。EAST网络结构图参考下图

![](../../../images/cv/text_detection/east/arch.jpg)

特征提取网络论文图中以PVANet网络为例，ppocr库中选用resnet50_vd以及mobilenet_v3作为特征提取网络。EAST算法输出层主要有三部分：

- socre map：特征融合层后接一个1*1的卷积，输出通道为1，最后输出一共分数图，代表每个像素点属于文本区域的概率。

- RBOX：这部分一共输出5个通道。分别由两个1*1卷积产生4个和1个，其中4个通道分别表示从像素位置到矩形的顶部，右侧，底部，左侧边界的4个距离，1个通道表示边界框的旋转角度。这部分用来预测旋转矩形的文本

- QUAD：使用8个数字来表示从四边形的四个角顶点{pi |i∈{1,2,3,4}}到像素位置的坐标偏移。由于每个距离偏移包含两个数字（Δxi，Δyi），因此几何输出包含8个通道。该部分可以预测不规则四边形的文本。

### pre-processing

EAST系列网络的预处理操作可以按照如下步骤进行，即先对图片(BGR)进行resize至[736, 1280]的尺寸（32的倍数），然后对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize((704, 1280)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### backbone

基于resnet50_vd或mobilenetv3等作为网络结构的骨干，分别从stage1，stage2，stage3，stage4的卷积层抽取出特征图，卷积层的尺寸依次减半，但卷积核的数量依次增倍，这是一种“金字塔特征网络”（FPN，feature pyramid network）的思想。通过这种方式，可抽取出不同尺度的特征图，以实现对不同尺度文本行的检测（大的feature map擅长检测小物体，小的feature map擅长检测大物体）。

### neck

将前面抽取的特征图按一定的规则进行合并，这里的合并规则采用了U-net方法，规则如下：

- 特征提取层中抽取的最后一层的特征图（f1）被最先送入unpooling层，将图像放大1倍
- 接着与前一层的特征图（f2）串起来（concatenate）
- 然后依次作卷积核大小为1x1，3x3的卷积
- 对f3，f4重复以上过程，而卷积核的个数逐层递减，依次为128，64，32
- 最后经过32核，3x3卷积后将结果输出到“输出层”


### post-processing

最终输出以下5部分的信息，分别是

- score map：检测框的置信度，1个参数；
- text boxes：检测框的位置（x, y, w, h），4个参数；
- text rotation angle：检测框的旋转角度，1个参数；
- text quadrangle coordinates：任意四边形检测框的位置坐标，(x1, y1), (x2, y2), (x3, y3), (x4, y4)，8个参数。

## Model Info

### 模型性能

| 模型  | 源码 | precision  | recall | Hmean | input size |
| :---: | :--: | :--: | :--: | :----: | :--------: |
|  det_r50_vd_east   | [ppocr](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/algorithm_det_east.md) | 88.71% | 81.36% |   84.88%   |    train 512x512 <br/> val 704×1280    |
|  det_mv3_east   | [ppocr](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/algorithm_det_east.md) | 78.20% | 79.10 |   78.65%   |    train 512x512 <br/> val 704×1280    |

> **Note**: 基于以下数据集
> 
> ICDAR 2015 ，train 512x512，val 704×1280



### 测评数据集说明


<div  align="center">
<img src="../../../images/dataset/icdar_2015.png" width="60%" height="60%">
</div>

[ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads)数据集包含1000张训练图像和500张测试图像。ICDAR 2015 数据集可以从上表中链接下载，首次下载需注册。 注册完成登陆后，下载下图中红色框标出的部分，其中， Training Set Images下载的内容保存在icdar_c4_train_imgs文件夹下，Test Set Images 下载的内容保存早ch4_test_images文件夹下。
```
train_data/icdar2015/text_localization/
  └─ icdar_c4_train_imgs/         icdar 2015 数据集的训练数据
  └─ ch4_test_images/             icdar 2015 数据集的测试数据
  └─ train_icdar2015_label.txt    icdar 2015 数据集的训练标注
  └─ test_icdar2015_label.txt     icdar 2015 数据集的测试标注
```

### 评价指标说明

- precision检测精度：正确的检测框个数在全部检测框的占比，主要是判断检测指标
- recall检测召回率：正确的检测框个数在全部标注框的占比，主要是判断漏检的指标
- hmean是前两项的调和平均值

## Build_In Deploy
- [ppocr](./source_code/ppocr.md)
