## ParseNet

## Code Source
```
link: https://github.com/yangxy/GPEN/blob/main/face_parse/parse_model.py
branch: master
commit: c9cc29009b633788a77d782ba102cee913e3a349
```


## Model Arch
<div  align="center">
<img src="../../../images/cv/segmentation/parsenet/parsenet.png" width="60%" height="60%">
</div>

### pre-processing

ParseNet模型的预处理操作可以按照如下步骤进行，即先对图片进行resize至一定尺寸(512)，然后对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(scale_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],),
]
```
### post-processing

ParseNet模型的后处理即是对网络输出的heatmap进行逐像素判断，比如一共19个类别，则网络会输出19个通道(18class+background)尺寸等于原图大小的heatmap，然后逐像素判断哪个通道数值大，就表示当前像素点所代表的类别为当前通道对应的类别。

### backbone
ParseNet模型的骨架网络由一系列的ResidualBlock组成的Encoder-Decoder全卷积结构（类似于unet）。

### common
- ResidualBlock
- Encoder-Decoder


## Model Info

## 模型精度

| Models | FLOPs(G) |Params(M) |  MIoU | Shapes|
|:-:|:-:|:-:|:-:|:-:|
| [ParseNet](https://github.com/yangxy/GPEN/blob/main/face_parse/parse_model.py) | 522.048 | 21.303 |  66.942 | 3×512×512|
| ParseNet **vacc_fp16**| - | - | 62.889 | 3×512×512|
| ParseNet **vacc_int8_kl_divergence**| - | - |62.721 | 3×512×512|

### 测评数据集说明

[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)是一个大规模的面部图像数据集，通过遵循CelebA-HQ从CelebA数据集中选择了30,000张高分辨率面部图像。 每个图像具有对应于CelebA的面部属性的分割MASK，其采用512 x 512尺寸手动标注，分为19类，包括所有面部组件和配件，例如皮肤，鼻子，眼睛，眉毛，耳朵，嘴巴，嘴唇，头发，帽子，眼镜，耳环，项链，脖子和布。CelebAMask-HQ可用于训练和评估人脸解析，人脸识别以及用于人脸生成和编辑的GAN的算法。

<div  align="center">
<img src="../../../images/dataset/celebamask-hq.png" width="60%" height="60%">
</div>


### 指标说明
- IoU并交比：两个区域重叠的部分除以两个区域的集合部分，取值TP/(TP+FN+FP)
- MIoU平均并交比：分割图像一般都有好几个类别，把每个分类得出的分数进行平均得到mean IoU，也就是mIoU，其是各种基准数据集最常用的标准之一，绝大数的图像语义分割论文中模型评估比较都以此作为主要评估指标。


## Build_In Deploy
- [gpen.md](./source_code/gpen.md)
