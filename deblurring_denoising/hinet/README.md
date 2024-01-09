
# HINet

[HINet: Half Instance Normalization Network for Image Restoration](https://arxiv.org/abs/2105.06086)

## Code Source
```
link: https://github.com/megvii-model/HINet
branch: main
commit: 4e7231543090e6280d03fac22b3bb6869a25dfad
```

## Model Arch

<div align=center><img src="../../images/hinet/arch.png"></div>

### pre-processing

HINet系列网络的预处理操作可以按照如下步骤进行：

```python
image = cv2.imread(image_file)
image_resize = cv2.resize(image, size)
img = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)

img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))).astype("float32") # HWC to CHW
img = img / 255.0
img = np.expand_dims(img, axis=0)
```

### post-processing

HINet模型后处理操作，按如下实现：
```python
heatmap = vacc_model.get_output(name, 0, 0).asnumpy().astype("float32")

output = np.squeeze(heatmap)
output = np.clip(output, 0, 1)
output = np.clip(output*255, 0, 255)
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
```

### backbone

本文整合一种新型的实例规范化（IN），用于构建基本模块，提高图像复原任务的网络性能。主要提出了一种半实例规范化模块（HIN block），并进而提出了一种多阶段网络（HINet）。通过将HIN block用于每个子网络的编码器中，提升特征的稳定性。此外，使用交叉阶段特征融合（CSFF）和监督注意力模块（SAM）[1]，用于丰富不同阶段网络之间的特征。

包括两个子网络，每个子网络是基于U-net。对于每个子网络而言，首先，利用3x3的卷积层提取基础特征；然后，这些特征被输入到具有4个下采样和上采样的编码解码器中；其中HIN block被整合到编码器的下采样模块中，用于提取特征，而在解码器中，使用ResBlocks用于提取高层次特征，并融合编码器的特征；最后，利用3x3卷积层获得用于重构图像的残差输出。
接着，利用交叉阶段特征融合（CSFF）和监督注意力模块（SAM）去连接两个子网络。

<div align=center><img src="../../images/hinet/hin_block.png"></div>

### common

- unet
- hin block
  
## Model Info

### 模型性能

| Models  | Flops(G) | Params(M) | PSNR(dB) | SSIM | Shape |
| :---: | :--: | :--: | :---: | :----: | :--------: |
| [HINet_SIDD_0.5x](https://github.com/megvii-model/HINet) |  95.283  |  22.175  |  39.676 | 0.920  |  3x256x256 |
| HINet_SIDD_0.5x **vacc fp16** |  -  |  -  |  38.190 |  0.913 |  3x256x256  |
| HINet_SIDD_0.5x **vacc percentile int8** |  -  |  -  |  29.274 | 0.885 |  3x256x256  |
| [HINet_SIDD_1x](https://github.com/megvii-model/HINet) |  379.365  |  88.666  |  39.546 | 0.921  |  3x256x256 |
| HINet_SIDD_1x **vacc fp16** |  -  |  -  |  38.588 |  0.918 |  3x256x256  |
| HINet_SIDD_1x **vacc percentile int8** |  -  |  -  |  37.517 | 0.901 |  3x256x256  |
| [HINet_GoPro](https://github.com/megvii-model/HINet) |  379.365  |  88.666  |  27.469 | 0.887  |  3x256x256 |
| HINet_GoPro **vacc fp16** |  -  |  -  |  27.418 |  0.885 |  3x256x256  |
| HINet_GoPro **vacc percentile int8** |  -  |  -  |   27.075 | 0.868 |  3x256x256  |
| [HINet_Rain13k](https://github.com/megvii-model/HINet) |  379.365  |  88.666  |  24.939 | 0.844  |  3x256x256 |
| HINet_Rain13k **vacc fp16** |  -  |  -  |  26.633 |  0.858 |  3x256x256  |
| HINet_Rain13k **vacc percentile int8** |  -  |  -  |   26.402 | 0.851 |  3x256x256  |


> Tips
>
> - 0.5x和1x对应网络深度不一样，0.5x模型更小
> - SIDD，GoPro，Rain13k对应不同数据集，后两个模型对应网络深度均为1x

### 测评数据集说明

[SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)数据集，全称 Smartphone Image Denoising Dataset，是一个图像降噪数据集。该数据集包括来自10个场景的约 3 万张噪声图像，由 5 个有代表性的智能手机摄像头拍摄，并生成了它们的 ground truth 图像。该数据集被用于来评估一些降噪算法。
<div  align="center">
<img src="../../images/dataset/sidd.jpg" width="70%" height="70%">
</div>


[GOPRO](https://seungjunnah.github.io/Datasets/gopro)数据集，作者使用GOPRO4 HERO Black相机拍摄了240fps的视频，然后对连续的7到13帧取平均获得模糊程度不一的图像。每张清晰图像的快门速度为1/240s，对15帧取平均相当于获得了一张快门速度为1/16s的模糊图像。作者将模糊图像对应的清晰图像定义为处于中间位置的那一帧图像。最终，一共生成了3214对模糊-清晰图像，分辨率为1280×720。

<div  align="center">
<img src="../../images/dataset/gopro.png" width="70%" height="70%">
</div>


### 评价指标说明
- 峰值信噪比(Peak Signal-to-Noise Ratio, PSNR)，PSNR是信号的最大功率和信号噪声功率之比，测量重构图像的质量，通常以分贝（dB）来表示。PSNR指标越高，说明图像质量越好
- 结构相似性评价(Structure Similarity Index, SSIM)，SSIM是衡量两幅图像相似度的指标，其取值范围为[0,1]，SSIM的值越大，表示图像失真程度越小，说明图像质量越好
- Fréchet Inception Distance，FID是衡量两个多元正态分布的距离，反映了生成图片和真实图片的距离，数据越小越好


## VACC部署
- [basicsr.md](./source_code/basicsr.md)
