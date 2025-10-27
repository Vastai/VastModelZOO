
# ECBSR

[Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices](https://www4.comp.polyu.edu.hk/~cslzhang/paper/MM21_ECBSR.pdf)

## Code Source
```
# official
link: https://github.com/xindongzhang/ECBSR
branch: main
commit: c60b1d2712af4ed4a615c2b7afa3980222c44a31

# basicsr
link: https://github.com/XPixelGroup/BasicSR
branch: v1.4.2
commit: 651835a1b9d38dbbdaf45750f56906be2364f01a
```

## Model Arch

<div align=center><img src="../../../images/cv/super_resolution/ecbsr/ecbsr.png"></div>

### pre-processing

ECBSR系列网络的预处理操作，可以按照如下步骤进行，首先尺寸缩放，然后转换至RGB，再转换至YCbCr，最后取Y通道，叠加成三通道（不同来源，预处理后处理不一致，具体参考推理脚本）：

```python
image_src = cv2.imread(image_file)
image_lr = cv2.resize(image_src, size) # , interpolation=cv2.INTER_AREA
image_lr = cv2.cvtColor(image_lr, cv2.COLOR_BGR2RGB)
ycbcr_lr = convert_rgb_to_ycbcr(image_lr.astype(np.float32))
image = ycbcr_lr[:,:,0]
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=0)
```

### post-processing

ECBSR系列网络的后处理操作，可以按照如下步骤进行：

```python
heatmap = model.get_output(name, 0, 0).asnumpy().astype("float32")
heatmap = np.squeeze(heatmap, axis=0)

# draw
ycbcr = cv2.resize(ycbcr_lr, (0,0), fx=2,fy=2)
# combine vacc output Y with source image cb and cy
sr = np.array([heatmap[0], ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
sr = np.clip(convert_ycbcr_to_rgb(sr), 0.0, 255.0)
output = sr.astype(np.uint8)[:,:,::-1]
```

### backbone

本文提出了称之为ECB的重参数化模块用于高效超分网络设计。在训练阶段，ECB采用多分支方式提取特征：包含常规3x3卷积、1x1卷积、一阶与二阶梯度信息；在推理阶段，上述多个并行操作可以折叠为单个3x3卷积。ECB可视作一种提升常规3x3卷积性能的“即插即用”模块且不会引入额外的推理消耗。我们基于ECB提出了一种用于端侧设备的超轻量超分网络ECBSR。

- 这个基本结构没有多分支也没有dense连接，因为这两种结构都有可能导致更高的内存访问成本，而且鉴于移动端设备的芯片带宽比较有限，这个基本拓扑结构只设计了一个残差连接，连接的是三通道的图像空间，而不是多通道的特征图空间。
- 基于结构重参数的思想，作者设计了一种模块ECB，这个模块可以更高效地提取图像的纹理信息和边缘信息，以便更好地完成SR任务，在训练时本文训练ECB的参数，但是在推理时，可以将ECB结构重参数化成一个3x3的卷积，提高推理速度。

ECB包含四个部分。
- 一个普通的3x3卷积，保证基本的性能。
- 扩展和挤压卷积，由一个1x1的卷积层和一个3x3的卷积层组成。1x1卷积扩展特征图维度（通道数），3x3卷积核再将特征图维度缩小挤压，最终的到特征信息更丰富的特征图。
- 一阶边缘信息提取，一条支路提取横向边缘，特征通过1x1卷积，然后通过一个尺度因子可学习的预先设定好的Sobel卷积核Dx。另一条支路提取纵向边缘，特征通过3x3卷积，然后通过一个尺度因子可学习的预先设定好的Sobel卷积核Dy。
- 二阶边缘特征提取，使用一个1x1卷积和一个尺度因子可学习的预先设定好的拉普拉斯卷积核

### common

- Edge-oriented Convolution Block
- Pixel-Shuffle

## Model Info

### 模型性能

| Models  |  Code Source |Flops(G) | Params(M) | PSNR(dB) | SSIM | Shape |
| :---: | :--: |:--: | :--: | :---: | :----: | :--------: |
| ecbsr_plain_2x | [Official](https://github.com/xindongzhang/ECBSR) |  12.478  |  0.0027  |  32.595 | 0.775 | 1x1080x1920 |
| ecbsr_plain_2x **vacc fp16** |  -  |  -  |  -  |  32.307 |  0.774 |  1x1080x1920  |
| ecbsr_plain_2x **vacc int8 percentile** |  -  |  -  |  -  |  32.344 |  0.769 |  1x1080x1920  |
| ecbsr_rgb_2x | [BasicSR](ttps://github.com/XPixelGroup/BasicSR/blob/master/docs/ModelZoo_CN.md) |  0.221  |   0.045  |  32.953 | 0.906 | 3x1080x1920 |
| ecbsr_rgb_2x **vacc fp16** |  -  |  -  |  -  |  32.953 |  0.906 |  3x1080x1920  |
| ecbsr_rgb_2x **vacc int8 max** |  -  |  -  |  -  |  31.797 |  0.852 |  3x1080x1920  |

> Tips
>
> - official来源，基于YCrCb颜色空间的Y通道进行训练，输入为单通道；未提供预训练权重，基于原始仓库训练代码，设置模型参数`sr_rate = 2`，重新训练`2x`模型；此处只提供重参数化后的plain模型
> - basics来源，基于[BasicSR](ttps://github.com/XPixelGroup/BasicSR)自训练而来，训练参数[train_ECBSR_x4_m4c16_prelu_RGB_mini.yml](source_code/basicsr/train_ECBSR_x4_m4c16_prelu_RGB_mini.yml)，输入为RGB
> - 精度指标基于DIV2K_valid_LR_bicubic/X2数据集


### 测评数据集说明

[DIV2K数据集](https://data.vision.ee.ethz.ch/cvl/DIV2K/)是一个受欢迎的单图像超分辨率数据集，可用于通过低分辨率图像重建高分辨率图像。
此数据集包含 1000 张具有不同退化类型的低分辨率图像，分为：
- 训练数据：800 张低分辨率图像，并为降级因素提供高分辨率和低分辨率图像。
- 验证数据：100 张高清高分辨率图片，用于生成低分辨率的图像。
- 测试数据：100 张多样化的图像，用来生成低分辨率的图像。

<div  align="center">
<img src="../../../images/dataset/div2k.png" width="70%" height="70%">
</div>



### 评价指标说明
- 峰值信噪比(Peak Signal-to-Noise Ratio, PSNR)，PSNR是信号的最大功率和信号噪声功率之比，测量重构图像的质量，通常以分贝（dB）来表示。PSNR指标越高，说明图像质量越好
- 结构相似性评价(Structure Similarity Index, SSIM)，SSIM是衡量两幅图像相似度的指标，其取值范围为[0,1]，SSIM的值越大，表示图像失真程度越小，说明图像质量越好
- Fréchet Inception Distance，FID是衡量两个多元正态分布的距离，反映了生成图片和真实图片的距离，数据越小越好


## Build_In Deploy
- [official.md](./source_code/official.md)
- [basicsr.md](./source_code/basicsr.md)

