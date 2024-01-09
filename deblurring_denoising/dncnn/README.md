
# DnCNN

[Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://ieeexplore.ieee.org/document/7839189)

## Code Source
```
link: https://github.com/cszn/DnCNN
branch: master
commit: e93b27812d3ff523a3a79d19e5e50d233d7a8d0a
```

## Model Arch

<div align=center><img src="../../images/dncnn/arch.png"></div>

### pre-processing

DnCNN系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后对其进行归一化等操作：

```python
def get_image_data(image_file, input_shape = [1, 1, 256, 256]):
    """fix shape resize"""
    size = input_shape[2:]

    src_image = cv2.imread(image_file, 0)
    image = cv2.resize(src_image, size)
    
    hr_image = copy.deepcopy(image)

    image = image.astype(np.float32) / 255.

    return image
```

### post-processing

DnCNN模型后处理操作，按如下实现：
```python
heatmap = vacc_model.get_output(name, 0, 0).asnumpy().astype("float32")

noise = np.squeeze(heatmap)
output = np.clip(output, 0, 1)
output = (output * 255.0).round().astype(np.uint8)
```

### backbone

DnCNN是图像去噪领域的经典模型，DnCNN在ResNet的基础上进行修改，网络结构是（卷积、BN、ReLU）级联的结构。不同的是DnCNN并非每隔两层就加一个shortcut connection，而是将网络的输出直接改成residual image（残差图片），设纯净图片为x，带噪音图片为y，假设y=x+v，则v是残差图片。即DnCNN的优化目标不是真实图片与网络输出之间的MSE(均方误差)，而是真实残差图片与网络输出之间的MSE。

网络结构：
- 第一部分：Conv（3 * 3 * C * 64）+ReLu
- 第二部分：Conv（3 * 3 * 64 * 64）+BN+ReLu
- 第三部分：Conv（3 * 3 * 64）

每一层都zero padding，使得每一层的输入、输出尺寸保持一致。以此防止产生人工边界（boundary artifacts）。第二部分每一层在卷积与ReLU之间都加了批量标准化（BN）。


### common

- Residual Learning
- ReLu
- BN
  
## Model Info

### 模型性能

| Models  | Flops(G) | Params(M) | PSNR(dB) | SSIM | Shape |
| :---: | :--: | :--: | :---: | :----: | :--------: |
| [DnCNN](https://github.com/cszn/DnCNN/tree/master/TrainingCodes/dncnn_pytorch) |  80.987  |  0.556  |  29.688 | 0.847  |  1x256x256 |
| DnCNN **vacc fp16** |  -  |  -  |  29.687 | 0.846 |  1x256x256  |
| DnCNN **vacc percentile int8** |  -  |  -  |  29.579 | 0.837 |  1x256x256  |



### 测评数据集说明

[Set12数据集](https://github.com/cszn/DnCNN/tree/master/testsets/Set12/)是数字图像处理的常用数据集，由12张灰度图组成（lena，cameraman，house，pepper，fishstar，monarch，airplane，parrot，barbara，ship，man，couple），01-07是256x256，08-12是512x512.

<div  align="center">
<img src="../../images/dataset/Set12.jpg" width="100%" height="100%">
</div>

### 评价指标说明
- 峰值信噪比(Peak Signal-to-Noise Ratio, PSNR)，PSNR是信号的最大功率和信号噪声功率之比，测量重构图像的质量，通常以分贝（dB）来表示。PSNR指标越高，说明图像质量越好
- 结构相似性评价(Structure Similarity Index, SSIM)，SSIM是衡量两幅图像相似度的指标，其取值范围为[0,1]，SSIM的值越大，表示图像失真程度越小，说明图像质量越好
- Fréchet Inception Distance，FID是衡量两个多元正态分布的距离，反映了生成图片和真实图片的距离，数据越小越好


## VACC部署
- [official.md](./source_code/official.md)
