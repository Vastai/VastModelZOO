
# Noise2Noise

[Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/abs/1803.04189)

## Code Source
```
# pytorch
link: https://github.com/joeylitalien/noise2noise-pytorch
branch: master
commit: 1a284a1a1c9db123e43b32e3f8bce277c5ca7b3b
```

## Model Arch


### pre-processing

Noise2Noise系列网络的预处理操作可以按照如下步骤进行：

```python
image = cv2.imread(image_file)
image_resize = cv2.resize(image, size)
img = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)

img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))).astype("float32") # HWC to CHW
img = img / 255.0
img = np.expand_dims(img, axis=0)
```

### post-processing

Noise2Noise模型后处理操作，按如下实现：
```python
heatmap = vacc_model.get_output(name, 0, 0).asnumpy().astype("float32")

output = np.squeeze(heatmap)
output = np.clip(output, 0, 1)
output = np.clip(output*255, 0, 255)
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
```

### backbone

该文章提出了一个很有意思的观点：在某些常见情况下，网络可以学习恢复信号而不用“看”到“干净”的信号，且得到的结果接近或相当于使用“干净”样本进行训练。而这项结论来自于一个简单的统计学上的观察：我们在网络训练中使用的损失函数，其仅仅要求目标信号（ground truth）在某些统计值上是“干净”的，而不需要每个目标信号都是“干净”的。
- 当网络有大量噪声到噪声的映射需要学习的时候，从loss最小化的角度来看，网络会输出所有目标噪声图像的均值。因为均值与各个目标图像的loss之和最小。
- 由于假设噪声是0均值的，因此可以说网络学会了去噪。
- N2N的缺点是需要大量的噪声图像对和噪声的0均值假设

### common

- unet
  
## Model Info

### 模型性能

| Models  | Code_source |Flops(G) | Params(M) | PSNR(dB) | SSIM | Shape |
| :---: | :--: | :--: |  :--: | :---: | :----: | :--------: |
| Noise2Noise |  [pytorch](https://github.com/joeylitalien/noise2noise-pytorch) | 41.447  |  0.700  |  37.151 | 0.900  |  3x256x256 |
| Noise2Noise **vacc fp16** |  -  |-  |  -  |  37.149 |  0.900 |  3x256x256  |
| Noise2Noise **vacc kl_divergence int8** |  -  | -  | -  |  36.820 | 0.896 |  3x256x256  |

> Tips
>
> - 基于SIDD/Val，sigma=25的gaussian加噪数据

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
