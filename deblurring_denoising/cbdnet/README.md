
# CBDNet

[Toward Convolutional Blind Denoising of Real Photographs](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR19-CBDNet.pdf)

## Code Source
```
link: https://github.com/IDKiro/CBDNet-pytorch
branch: master
commit: 09a2e55b2098039ee99ada8c634a06fc28c6d8a1
```


## Model Arch

<div align=center><img src="../../images/cbdnet/arch.png"></div>

### pre-processing

CBDNet系列网络的预处理操作可以按照如下步骤进行：

```python
image = cv2.imread(image_file)
image_resize = cv2.resize(image, size)
img = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)

img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))).astype("float32") # HWC to CHW
img = img / 255.0
img = np.expand_dims(img, axis=0)
```

### post-processing

CBDNet模型后处理操作，按如下实现：
```python
heatmap = vacc_model.get_output(name, 0, 0).asnumpy().astype("float32")

output = np.squeeze(heatmap)
output = np.clip(output, 0, 1)
output = np.clip(output*255, 0, 255)
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
```

### backbone

针对真实图像的卷积盲去噪，Convolutional Blind Denoising，卷积盲去噪网络CBDNet由噪声估计子网络(FCN)和非盲去噪子网络(UNet)两部分组成。
- 提出了一个更加真实的噪声模型，既考虑了Poisson-Gaussian model，还考虑了信号依赖噪声和ISP对噪声的影响
- 提出了CBDNet，包括一个噪声估计子网络和一个非盲去噪子网络，可以实现图像的盲去噪(即未知噪声水平)
- 将合成噪声图像(synthetic noisy images)和真实噪声图像(real-world noisy images)结合起来训练网络，以更好地表述real-world图像的噪声，提高去噪性能
- 因为噪声估计网络的存在，引入了一个非对称损失(asymmetric loss)，进一步增强了去噪性能和泛化能力。并且可以通过调整噪声水平图来实现交互式去噪

### common

- unet
- convTranspose2d
  
## Model Info

### 模型性能

| Models  | Flops(G) | Params(M) | PSNR(dB) | SSIM | Shape |
| :---: | :--: | :--: | :---: | :----: | :--------: |
| [CBDNet](https://github.com/IDKiro/CBDNet-pytorch) |  89.519  |  4.365  |  38.262 | 0.903  |  3x256x256 |
| CBDNet **vacc fp16** |  -  |  -  | 38.262 |  0.903 |  3x256x256  |
| CBDNet **vacc percentile int8** |  -  |  -  |  37.798 | 0.895 |  3x256x256  |



### 测评数据集说明

[SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)数据集，全称 Smartphone Image Denoising Dataset，是一个图像降噪数据集。该数据集包括来自10个场景的约 3 万张噪声图像，由 5 个有代表性的智能手机摄像头拍摄，并生成了它们的 ground truth 图像。该数据集被用于来评估一些降噪算法。
<div  align="center">
<img src="../../images/dataset/sidd.jpg" width="70%" height="70%">
</div>


### 评价指标说明
- 峰值信噪比(Peak Signal-to-Noise Ratio, PSNR)，PSNR是信号的最大功率和信号噪声功率之比，测量重构图像的质量，通常以分贝（dB）来表示。PSNR指标越高，说明图像质量越好
- 结构相似性评价(Structure Similarity Index, SSIM)，SSIM是衡量两幅图像相似度的指标，其取值范围为[0,1]，SSIM的值越大，表示图像失真程度越小，说明图像质量越好
- Fréchet Inception Distance，FID是衡量两个多元正态分布的距离，反映了生成图片和真实图片的距离，数据越小越好


## VACC部署
- [pytorch.md](./source_code/pytorch.md)
