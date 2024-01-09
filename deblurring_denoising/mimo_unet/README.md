
# MIMO_UNet

[Rethinking Coarse-to-Fine Approach in Single Image Deblurring](https://arxiv.org/abs/2108.05054)

## Code Source
```
link: https://github.com/chosj95/mimo-unet
branch: main
commit: 5c580135ed1c03344ac9c741267324ff90b5f209
```

## Model Arch

<div align=center><img src="../../images/mimo_unet/arch.jpg"></div>

### pre-processing

MIMO_UNet系列网络的预处理操作可以按照如下步骤进行：

```python
image = cv2.imread(image_file)
image_resize = cv2.resize(image, size)
img = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)

img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))).astype("float32") # HWC to CHW
img = img / 255.0
img = np.expand_dims(img, axis=0)
```

### post-processing

MIMO_UNet模型后处理操作，按如下实现：
```python
heatmap = vacc_model.get_output(name, 0, 0).asnumpy().astype("float32")

output = np.squeeze(heatmap)
output = np.clip(output, 0, 1)
output += 0.5 / 255 # same as torch
output = np.clip(output*255, 0, 255)
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
```

### backbone

本文重新思考了Coarse-to-Fine图像去模糊方法，提出了多输入多输出的U-net（MIMO-UNet）结构。编码器输入多尺度模糊图像，解码器输出相应尺度的模糊修复图像。此外，提出了非对称特征融合用于结合多尺度的特征。

- 首先，本文网络MIMO-UNet的骨干就是个UNet。为了可以让更小尺度的模糊图可以顺利输入到网络中，它们经过SCM编码后送入FAM（绿块）。
- 然后，为了让Encoder中三个不同大小层的信息（feature map）可以融合，将它们通过downsample or upsample后使用AFF融合，输入到decoder中的同层。
- 最后，decoder的结果经过一层卷积 & 与原图的残差连接，即可获得deblur图。

### common

- unet
- shallow convolutional module(SCM)
  
## Model Info

### 模型性能

| Models  | Flops(G) | Params(M) | PSNR(dB) | SSIM | Shape |
| :---: | :--: | :--: | :---: | :----: | :--------: |
| [MIMO_UNet](https://github.com/chosj95/mimo-unet) |  523.079  |  6.807  |  30.302 | 0.914  |  3x360x640 |
| MIMO_UNet **vacc fp16** |  -  |  -  |  29.857 |  0.906 |  3x360x640  |
| MIMO_UNet **vacc percentile int8** |  -  |  -  |  29.274 | 0.885 |  3x360x640  |
| [MIMO_UNetPlus](https://github.com/chosj95/mimo-unet) |  1202.556  |  16.108  |  30.853 | 0.924  |  3x360x640 |
| MIMO_UNet **vacc fp16** |  -  |  -  |  30.606 |  0.920 |  3x360x640  |
| MIMO_UNet **vacc percentile int8** |  -  |  -  |  29.379 | 0.887 |  3x360x640  |


> Tips
>
> - 源模型中，上采样和下采样操作[F.interpolate](https://github.com/chosj95/MIMO-UNet/blob/main/models/MIMOUNet.py#L140)使用默认的nearest方式，在vdps中存在尺寸限制(3x360x640 int8 run 会报错)，替换为mode='bilinear'导出模型，可避免尺寸限制
> - 3x720x1280尺寸下，fp16 run 会超显存，int8 run ok
> - MIMO_UNet和MIMO_UNetPlus的区别在于ResBlock的数量不同，前者为8，后者为20

### 测评数据集说明

[GOPRO](https://seungjunnah.github.io/Datasets/gopro)数据集，作者使用GOPRO4 HERO Black相机拍摄了240fps的视频，然后对连续的7到13帧取平均获得模糊程度不一的图像。每张清晰图像的快门速度为1/240s，对15帧取平均相当于获得了一张快门速度为1/16s的模糊图像。作者将模糊图像对应的清晰图像定义为处于中间位置的那一帧图像。最终，一共生成了3214对模糊-清晰图像，分辨率为1280×720。

<div  align="center">
<img src="../../images/dataset/gopro.png" width="70%" height="70%">
</div>


### 评价指标说明
- 峰值信噪比(Peak Signal-to-Noise Ratio, PSNR)，PSNR是信号的最大功率和信号噪声功率之比，测量重构图像的质量，通常以分贝（dB）来表示。PSNR指标越高，说明图像质量越好
- 结构相似性评价(Structure Similarity Index, SSIM)，SSIM是衡量两幅图像相似度的指标，其取值范围为[0,1]，SSIM的值越大，表示图像失真程度越小，说明图像质量越好
- Fréchet Inception Distance，FID是衡量两个多元正态分布的距离，反映了生成图片和真实图片的距离，数据越小越好


## VACC部署
- [official.md](./source_code/official.md)
