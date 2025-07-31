
# RCAN

[Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758)

## Code Source
```
# official
link: https://github.com/yulunzhang/RCAN
branch: master
commit: 3339ebc59519c3bb2b5719b87dd36515ec7f3ba7

# basicsr
link: https://github.com/XPixelGroup/BasicSR
branch: master
commit: 033cd6896d898fdd3dcda32e3102a792efa1b8f4
```

## Model Arch

<div align=center><img src="../../../images/cv/super_resolution/rcan/arch.png"></div>

### pre-processing

RCAN系列网络的预处理操作，可以按照如下步骤进行（不同来源预处理和后处理可能不同，实际请参考对应推理代码）：

```python
def get_image_data(image_file, input_shape = [1, 3, 1080, 1920]):
    size = input_shape[2:][::-1]

    image = cv2.imread(image_file)
    img = cv2.resize(image, size) # , interpolation=cv2.INTER_AREA
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) # HWC to CHW
    img = np.expand_dims(img, axis=0)

    return np.array(img)
```

### post-processing

RCAN系列网络的后处理操作，可以按照如下步骤进行：
```python
heatmap = vacc_model.get_output(name, 0, 0).asnumpy().astype("float32")

output = np.squeeze(heatmap)
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
```

### backbone

卷积神经网络（CNN）深度是图像超分辨率（SR）的关键。然而，我们观察到图像SR的更深的网络更难训练。低分辨率的输入和特征包含了丰富的低频信息，这些信息在不同的信道中被平等地对待，从而阻碍了CNNs的表征能力。为了解决这些问题，我们提出了超深剩余信道注意网络（RCAN）。具体地说，我们提出了一种残差中残差（RIR）结构来形成非常深的网络，它由多个具有长跳跃连接的残差组组成。每个剩余组包含一些具有短跳过连接的剩余块。同时，RIR允许通过多跳连接绕过丰富的低频信息，使主网集中学习高频信息。此外，我们提出了一种通道注意机制，通过考虑通道间的相互依赖性，自适应地重新缩放通道特征。RCAN与现有的方法相比，具有更好的精确度和视觉效果。

网络有四个部分：浅层特征提取，RIR 深层特征提取，上采样模块，重建部分。
只使用第一层卷积来提取浅层特征，接着会在使用RIR模块的深层特征提取，RIR有更大的depth和提供了更大的感受野，所以叫做deep feature，最后做上采样（pixshuffle）。

- Residual in Residual

    RIR架构中，RG（Residual Group）作为基本模块，LSC（Long Skip Connection）则用来进行粗略的残差学习，在每个 RG 内部则叠加数个简单的残差块和 SSC（Short Skip Connection）。LSC、SSC 和残差块内部的短连接可以允许丰富的低频信息直接通过恒等映射向后传播，这可以保证信息的流动，加速网络的训练。

- Channel Attention
  
    提出了一个信道注意机制来自适应调整信道特征，通过考虑信道间的相互依赖性。主要有两个地方：首先，LR空间中的信息具有丰富的低频成分和有价值的高频成分，低频部分似乎更加扁平化，高频成分通常是区域，充满边缘，纹理和其他细节。另一方面，Conv层中的每个滤波器都以局部感受野运行。因此，卷积后的输出无法利用本地区域之外的上下文信息。
    <div  align="center">
    <img src="../../../images/cv/super_resolution/rcan/ca.png" width="90%" height="90%">
    </div>

    如上图所示，输入是一个 H×W×C 的特征，我们先进行一个空间的全局平均池化得到一个 1×1×C 的通道描述。接着，再经过一个下采样层和一个上采样层得到每一个通道的权重系数，将权重系数和原来的特征相乘即可得到缩放后的新特征，整个过程实际上就是对不同通道的特征重新进行加权分配。
    其中，下采样和上采样层都利用 1×1 的卷积来实现，下采样层的通道数减少r倍，激活函数为 Relu，上采样层的激活函数为 Sigmoid。在论文中，作者采用的通道数 C=64，r = 16。
    
    最后作者把Residual in residual和Channel Attention 结合到一起组成了RCAB模块作为RCAN的基础模块。
    <div  align="center">
    <img src="../../../images/cv/super_resolution/rcan/rcab.png" width="90%" height="90%">
    </div> 

### common

- RIR
- Channel Attention
- Pixel-Shuffle

## Model Info

### 模型性能

| Models  |  Code Source |Flops(G) | Params(M) | PSNR(dB) | SSIM | Shape |
| :---: | :--: |:--: | :--: | :---: | :----: | :--------: |
| RCAN | [Official](https://github.com/yulunzhang/RCAN) |  358.207  |  0.297  |  32.785 | 0.776 |  -  |
| RCAN **vacc max int8** |  -  |  -  |  -  |  32.628 | 0.766 |  3x1080x1920  |
| RCAN2 | [Official](https://github.com/yulunzhang/RCAN) |  80.331  |  0.063  |  32.917 | 0.775 |  3x1080x1920  |
| RCAN2 **vacc max int8** |  -  |  -  |  -  |  32.349 | 0.764 |  3x1080x1920  |
| RCAN | [BasicSR](https://github.com/XPixelGroup/BasicSR) |  356.659  |  0.298  |  29.470 | 0.731 |  3x1080x1920  |
| RCAN **vacc max int8** |  -  |  -  |  -  |  29.551 | 0.738 |  3x1080x1920  |

> Tips
>
> - RCAN，Official来源模型，基于原始模型[rcan.py](https://github.com/yulunzhang/RCAN/blob/master/RCAN_TestCode/code/model/rcan.py)进行修改：去除了通道注意力等模块；减少通道深度及block堆叠数量(--n_resgroups 3 --n_resblocks 2 --n_feats 32)，以精简计算量
> - RCAN2，Official来源模型，在以上精简基础上进一步减少(--n_resgroups 2 --n_resblocks 2 --n_feats 16)，加大训练时间和crop尺寸192
>
> - 内部定义此模型为: SR4K
> - 因有PixelShuffle，fp16 vacc暂不支持
> - 精度指标基于DIV2K valid两倍放大数据集


### 测评数据集说明

[DIV2K数据集](https://data.vision.ee.ethz.ch/cvl/DIV2K/)是一个受欢迎的单图像超分辨率数据集，可用于通过低分辨率图像重建高分辨率图像。
此数据集包含 1000 张具有不同退化类型的低分辨率图像，分为：
- 训练数据：800 张低分辨率图像，并为降级因素提供高分辨率和低分辨率图像。
- 验证数据：100 张高清高分辨率图片，用于生成低分辨率的图像。
- 测试数据：100 张多样化的图像，用来生成低分辨率的图像。

<div  align="center">
<img src="../../../images/dataset/div2k.png" width="70%" height="70%">
</div>


[Set5](https://github.com/twtygqyy/pytorch-vdsr/tree/master/Set5)数据集是基于非负邻域嵌入的低复杂度单图像超分辨率的数据集（共5张BMP图像），该训练集被用于单幅图像超分辨率重构，即根据低分辨率图像重构出高分辨率图像以获取更多的细节信息。
<div  align="center">
<img src="../../../images/dataset/Set5.png" width="70%" height="70%">
</div>


### 评价指标说明
- 峰值信噪比(Peak Signal-to-Noise Ratio, PSNR)，PSNR是信号的最大功率和信号噪声功率之比，测量重构图像的质量，通常以分贝（dB）来表示。PSNR指标越高，说明图像质量越好
- 结构相似性评价(Structure Similarity Index, SSIM)，SSIM是衡量两幅图像相似度的指标，其取值范围为[0,1]，SSIM的值越大，表示图像失真程度越小，说明图像质量越好
- Fréchet Inception Distance，FID是衡量两个多元正态分布的距离，反映了生成图片和真实图片的距离，数据越小越好


## Build_In Deploy
- [official.md](./source_code/official.md)
- [basicsr.md](./source_code/basicsr.md)
