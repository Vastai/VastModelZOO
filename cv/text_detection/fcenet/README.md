# FCENet
[Fourier Contour Embedding for Arbitrary-Shaped Text Detection](https://arxiv.org/abs/2104.10442)

## code source

```
# mmocr
link: https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/fcenet
branch: main
commit: b18a09b2f063911a2de70f477aa21da255ff505d
```

## Model Arch

FCENet (Fourier Contour Embedding for Arbitrary-Shaped Text Detection) 通过预测一种基于傅里叶变换的任意形状文本包围框表示，从而实现了自然场景文本检测中对于高度弯曲文本实例的检测精度的提升

### pre-processing

FCENet系列网络的预处理操作可以按照如下步骤进行，即先对图片(BGR)进行resize至对应的尺寸（32的倍数），然后对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize((736, 1280)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### backbone

FCENet 的算法流程如图所示，输入图像会经过主干网络和 FPN 提取特征。FPN 上不同层负责不同尺度的文本目标，提取的特征会送入到共享的检测头中。共享的检测头具有两个分支，其中分类分支预测文本区域和文本中心区域的概率图，相乘得到属于文本中心分类的得分图；回归分支则负责预测傅里叶特征向量。算法对文本中心置信度大于阈值的像素点所对应的傅里叶特征向量进行傅里叶反变换，并经过非极大值抑制得到最终的检测结果。

![](../../../images/cv/text_detection/fcenet/arch.png)


### post-processing

FCENet算法后处理对文本中心置信度大于阈值的像素点所对应的傅里叶特征向量进行傅里叶反变换，并经过非极大值抑制得到最终的检测结果。

## Model Info

### 模型性能

| 模型  | 源码 | precision  | recall | f-measure | input size | datasets |
| :---: | :--: | :--: | :--: | :----: | :--------: | :--------:|
|  fcenet_resnet50_fpn_1500e_icdar2015   | [mmocr](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/fcenet) | 82.43 | 88.34 |   85.28   | 736x1280 | icdar2015 |
|  fcenet_resnet50_oclip_fpn_1500e_icdar2015   | [mmocr](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/fcenet) | 91.76 | 80.98 |   86.04   | 736x1280 | icdar2015 |
|  fcenet_resnet50_fpn_1500e_totaltext   | [mmocr](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/fcenet) | 84.85 | 78.10 |   81.34   | 960x1280 | total-text |
|  fcenet_resnet50_oclip_fpn_1500e_ctw1500   | [mmocr](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/fcenet) | 83.83 | 80.1 |   81.92   | 736x1280 | ctw1500 |


- 上述表中尺寸即为modelzoo提供模型的尺寸，精度为原始mmocr库中提供的尺寸


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

- [mmocr](./source_code/mmocr.md)
