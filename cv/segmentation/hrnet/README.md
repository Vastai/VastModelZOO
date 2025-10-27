## hrnet

[Deep High-Resolution Representation Learning
for Visual Recognition](https://arxiv.org/abs/1908.07919)


## Code Source
```
link: https://github.com/HRNet/HRNet-Semantic-Segmentation
branch: pytorch-v1.1
commit: 88419ab18813f2c9193985e2d4d31d3d07abe839
```


## Model Arch

<div  align="center">
<img src="../../../images/cv/segmentation/hrnet_seg/arch.png">
</div>

### pre-processing

HRNet网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至一定尺寸(512)，然后对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(scale_size),
    torchvision.transforms.CenterCrop(input_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

HRNet算法的后处理即是对网络输出的heatmap进行逐像素判断，比如一共20个类别，则网络会输出21个通道(20class+background)尺寸等于原图大小的heatmap，然后逐像素判断哪个通道数值大，就表示当前像素点所代表的类别为当前通道对应的类别

### backbone
HRNet作为主干网络提取了特征，这些特征有不同的分辨率，需要根据不同的任务来选择融合的方式。

HRNet的设计思路延续了一路保持较大分辨率特征图的方法，在网络前进的过程中，都**保持较大的特征图**，但是在网路前进过程中，也会**平行地**做一些下采样缩小特征图，如此**迭代**下去。最后生成**多组有不同分辨率的特征图**，**再融合**这些特征图做Segmentation map的预测。

### common

- Multi-resolution Fusion

## Model Info

### 模型精度

|                  Models                  |                                   Code Source                                   | GFLOPs(G) | Params(M) |  mIOU  |   Shapes   |
| :---: | :--: | :--: | :--: | :---: | :--------: |
| HRNetV2-W48-Cityscapes |[official](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1)|  696.2   |   65.8    |      81.1   | 3x1024x2048    |
| HRNetV2-W48-Cityscapes |-|   -   |   -    |    65.8   |  3x512x512   |
| HRNetV2-W48-Cityscapes-vacc-fp16 |-|    -   |   -    |     62.215   | 3x512x512   |
| HRNetV2-W48-Cityscapes-vacc-int8 |-|   -   |   -    |      63.754   | 3x512x512   |
| HRNetV2-W18-Small-v1-Cityscapes |[official](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1)|    31.1   |   1.5    |      70.3   | 3x1024x2048    |
| HRNetV2-W18-Small-v1-Cityscapes |-|    -   |  -    |    58.6   | 3x512x512    |
| HRNetV2-W18-Small-v2-Cityscapes-vacc-fp16 |-|    -   |   -    |      54.482   | 3x512x512    |
| HRNetV2-W18-Small-v2-Cityscapes-int8 |-|    -   |   -    |      53.814  | 3x512x512    |
| HRNetV2-W18-Small-v2-Cityscapes |[official](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1)|     71.6   |   3.9    |     76.2   | 3x1024x2048    |
| HRNetV2-W18-Small-v2-Cityscapes |-|     -   |   -    |     60.4   | 3x512x512    |
| HRNetV2-W18-Small-v2-Cityscapes-vacc-fp16 |-|     -   |   -    |    58.037  | 3x512x512    |
| HRNetV2-W18-Small-v2-Cityscapes-vacc-int8 |-|     -   |   -    |     58.881   | 3x512x512    |


### 测评数据集说明

[CityScapes](https://www.cityscapes-dataset.com/)数据集，即城市景观数据集，这是一个新的大规模数据集，其中包含不同的立体视频序列，记录在50个不同城市的街道场景。数据集被分为2975 train，500 val，1525 test，它具有19个类别的密集像素标注。

<div  align="center">
<img src="../../../images/dataset/cityscapes.png" width="80%" height="80%">
</div>


### 指标说明
- IoU并交比：两个区域重叠的部分除以两个区域的集合部分，取值TP/(TP+FN+FP)
- MIoU平均并交比：分割图像一般都有好几个类别，把每个分类得出的分数进行平均得到mean IoU，也就是mIoU，其是各种基准数据集最常用的标准之一，绝大数的图像语义分割论文中模型评估比较都以此作为主要评估指标。


## Build_In Deploy
- [official.md](./source_code/official.md)
