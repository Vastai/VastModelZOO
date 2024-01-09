
# IDR

[IDR: Self-Supervised Image Denoising via Iterative Data Refinement](https://arxiv.org/abs/2111.14358)

## Code Source
```
link: https://github.com/zhangyi-3/IDR
branch: main
commit: f3f05d4bceff6ab780a841c6997c5daead859bda
```

## Model Arch

<div align=center><img src="../../images/idr/arch.png"></div>

### pre-processing

IDR系列网络的预处理操作可以按照如下步骤进行：

```python
image = cv2.imread(image_file)
image_resize = cv2.resize(image, size)
img = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)

img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))).astype("float32") # HWC to CHW
img = img / 255.0
img = img - 0.5
img = np.expand_dims(img, axis=0)
```

### post-processing

IDR模型后处理操作，按如下实现：
```python
heatmap = vacc_model.get_output(name, 0, 0).asnumpy().astype("float32")

output = np.squeeze(heatmap)
output += 0.5
output = np.clip(output*255, 0, 255)
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
```

### backbone

虽然现有的无监督方法能够学习图像去噪，而无需真实的干净图像，但它们要么表现出较差的性能，要么在不切实际的设置下工作（如成对的噪声图、只针对加性噪声）。

本文提出了一种自监督图像去噪方案——迭代数据细化（IDR）。只需要单个噪声图像和噪声模型，就可以实现较好的去噪效果，对真实噪声、合成和相关噪声的实验表明，提出的无监督去噪方法比现有的无监督方法具有更好的性能，并且与有监督方法具有竞争性。

快速迭代算法：
- 每个数据集只训练一个epoch，我们牺牲了完整模型优化所需的时间，但增加了数据优化的迭代次数。它的成本不到总训练时间的5%。因此，总训练时间减少到与仅训练一轮去噪模型的时间几乎相同。
- 当在每个时期对新数据集进行训练时，我们的模型由上一时期的模型初始化，这种累积训练策略有助于去噪网络通过所提出的快速数据细化方案更快地收敛，并确保在整个训练过程中不断优化最终的去噪模型。



### common

- unet
  
## Model Info

### 模型性能

| Models  | Flops(G) | Params(M) | PSNR(dB) | SSIM | Shapes |
| :---: | :--: | :--: | :---: | :----: | :--------: |
| [idr_gaussian](https://github.com/zhangyi-3/IDR) |  25.351  |  0.991  | 37.013 | 0.896  |  3x256x256 |
| idr_gaussian **vacc fp16** |  -  |  -  |  37.015 |  0.896 |  3x256x256  |
| idr_gaussian **vacc percentile int8** |  -  |  -  |  36.389 | 0.888 |  3x256x256  |



> Tips
>
> - 基于SIDD数据集，gaussian sigma=25合成加噪数据上验证

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
- [official.md](./source_code/official.md)
