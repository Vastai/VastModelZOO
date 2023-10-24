
# ISNet

[Highly Accurate Dichotomous Image Segmentation](https://arxiv.org/abs/2203.03041)

## Code Source
```
link: https://github.com/xuebinqin/DIS
branch: main
commit: f3837183a33dab157c636e0124e091acd6da9dd1
```

## Model Arch

### pre-processing

ISNet系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至一定尺寸，然后对其进行归一化等操作：

```python
image = cv2.imread(image_file)
img = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mean = np.array([0.5, 0.5, 0.5])
std  = np.array([1.0, 1.0, 1.0])
img = (img/255.0 - mean) / std
```

### post-processing

ISNet系列网络的后处理操作，主要有反归一化：
```python
mask = heatmap[0][0]
mask = Image.fromarray(mask*255).convert('RGB')
```

### backbone

ISNet由一个真值(GT)编码器、一个图像分割组件和一个新提出的中间监督策略组成。
- GT编码器用于将GT掩码编码到高维空间，然后用于对分割组件进行中间监督
- 图像分割组件被期望在可承受的内存和时间成本下，具有捕获精细结构并处理大尺寸（比如 1024 × 1024）输入的能力。本文选择 U2-Net作为图像分割组件，因为它对精细结构的捕捉能力较强
- 中间监督，DIS可以看作是分割模型中从图像域到分割GT域的映射。大多数模型在训练集上容易过度拟合。因此，深度监督被提出来对给定的深度网络的中间输出进行监督。密集监督通常应用于侧输出，侧输出是通过卷积特定深度层的最后一层特征图而产生的单通道概率图

<div  align="center">
<img src="../../images/isnet/is-net.png" width="80%" height="80%">
</div>

### common

- GT Encoder
- Intermediate Supervision

## Model Info

### 模型性能

| Models  | Flops(G) | Params(M) | MAE ↓ | avg F-Measure ↑ | SM ↑ | Shapes |
| :---: | :--: | :--: | :---: | :--------: | :---: | :--------: |
| [ISNet](https://github.com/xuebinqin/DIS) | 34.753  |  44.047 | 0.114  |  0.672  | 0.789  | 3x320x320  |
| ISNet **vacc fp16** |  -  |  -  |  0.115  |  0.671  | 0.788  | 3x320x320  |
| ISNet **vacc kl_divergence int8** |  -  |  -  |   0.116  | 0.667 | 0.786 |  3x320x320  |


### 测评数据集说明


[ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)数据集，由香港中文大学的Yan等人于2013年建立, 包含了1000张图像, 这些图像由互联网得到。该数据集中的显著物体包含较复杂的结构, 且背景具备一定的复杂性。


<div  align="center">
<img src="../../images/datasets/ecssd.jpg" width="80%" height="70%">
</div>

### 评价指标说明
显著性目标检测主要的评测指标包括：
- 均值绝对误差（Mean Absolute Error，MAE），用于通过测量归一化映射和真值掩码之间平均像素方向的绝对误差来解决这个问题，越小越好
- EMD距离(earth movers distance，EMD)，衡量的是显著性预测结果P与连续的人眼注意力真值分布Q之间的相似性, 该度量方式被定义为:从显著性预测结果P上的概率分布转移到连续的人眼注意力真值分布Q上的最小代价。因而, EMD距离越小, 表示估计结果越准确
- 交叉熵(kullback-leibler divergence，KLD)，主要基于信息理论, 经常被用于衡量两个概率分布之间的距离，在人眼关注点检测中, 该指标被定义为:通过显著性预测结果P来近似连续的人眼注意力真值分布Q时产生的信息损失，越小越好
- 标准化扫描路径显著性(normalized scanpath saliency, NSS)，是专门为显著性检测设计的评估指标，该指标被定义为:对在人眼关注点位置归一化的显著性(均值为0和归一化标准差)求平均。越小越好
- 线性相关系数(linear correlation coefficient, CC)，是一种用于衡量两个变量之间相关性的统计指标，在使用该度量时, 将显著性预测结果P和连续的人眼注意力真值分布Q视为随机变量。然后, 统计它们之间的线性相关性。该统计指标的取值范围是[-1, +1].当该指标的值接近-1或+1时, 代表显著性预测结果与真值标定高度相似
- 相似性测度(similarity metric, SIM)指标，将显著性预测结果P和连续的人眼注意力真值分布Q视为概率分布, 将二者归一化后, 通过计算每一个像素上的最小值, 最后加和得到。当相似性测度为1时, 表示两个概率分布一致; 为0时, 表示二者完全不同
- AUC指标(the area under the receiver operating characteristic curve, 简称ROC曲线), 即受试者工作特性曲线下面积.ROC曲线是以假阳性概率(false positive rate, FPR)为横轴, 以真阳性概率(true positive rate, 简称TPR)为纵轴所画出的曲线。AUC即为ROC曲线下的面积, 通过在[0, 1]上滑动的阈值, 能够将显著性检测结果P进行二值化, 从而得到ROC曲线。ROC曲线越趋近于左上方, AUC数值越大, 说明算法性能越好。当接近1时, 代表着显著性估计与真值标定完全一致
- F-Measure，由于查准率和查全率相互制约, 且查准率-查全率曲线包含了两个维度的评估指标, 不易比较, 因而需要就二者进行综合考量。该指标同时考虑了查准率和查全率, 能够较为全面、直观地反映出算法的性能。F-值指标的数值越大, 说明算法性能越好
- 结构相似性（Structural measure，S-measure）：用以评估实值显著性映射与真实值之间的结构相似性，其中So和Sr分别指对象感知和区域感知结构的相似性，越大越好

## VACC部署
- [official.md](./source_code/official.md)