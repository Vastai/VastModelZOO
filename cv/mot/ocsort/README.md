# OCSORT

## Code Source
```
link: https://github.com/noahcao/OC_SORT/tree/master
branch: master
commit: 7a390a5f35dbbb45df6cd584588c216aea527248
```

## Model Arch

这篇文章解决的主要问题是，现有的方法对运动预测都是基于线性运动假设，对非线性运动、遮挡、低帧率视频就没有好的处理效果。

作者说SORT有三个缺点：

1. 帧率很高的情况，目标位移的噪声可能就和位移大小本身差不多（因为位移会很小），这样Kalman的方差会很大。
2. 由于遮挡等原因，如果没有新的轨迹和检测匹配，那目标的噪声可能会积累。证明了误差积累关于时间是平方关系。
3. Kalman主要依赖于状态估计，把观测只作为辅助信息。作者认为现在的检测器足够强，应该已经可以把观测作为主要信息了（Observation-Centirc的来源）

针对以上三个问题，作者提出了三个创新：

1. 提出了observation为中心的平滑策略，来减少误差累积。具体地就是一个inactive的轨迹要Re-ID的时候，先为这个轨迹建立一个虚拟轨迹。虚拟轨迹的起始为它上一次被发现的位置，终止为它再次被发现的位置。在这两个位置之间作一个平滑。
2. 在cost matrix中加入了轨迹方向的一致性，证明了对相隔比较大的两点之间的方向进行估计可以减少噪声。
3. 为了减少在短时内目标因遮挡成为inactive的问题，作者提出了根据它们新就旧观测进行恢复的策略。

### 跟踪算法流程

![](../../../images/cv/mot/ocsort/teaser.png)

1. 以观测为中心的在线平滑 OOS

    当一个轨迹（一段时间跟丢了）再次与观测关联上之后，通过观测的虚拟轨迹，在线平滑参数，使其回到丢失的时刻，这可以解决这段时间内的累积误差。

2. 观察中心动量OCM

    线性运动模型假设速度方向一致，但在现实中由于目标的非线性运动和噪声，这种假设时不存在。在相对短的时间内，我们可以吧运动近似于线性，但是噪声仍然会影响速度方向的一致性。提出了一种方式来减少噪声，在代价矩阵中添加了速度一致性（动量）项。

3. 观察为中心的恢复

    一般来说，轨迹跟丢通常是由于观测丢失（遮挡或不可靠的检测）和非线形运动造成的。在以观测为中心的观念中，保守策略是在跟丢的位置附近搜索（随着时间推移类似高斯分布，以跟丢位置为中心，方差随着时间而放大）。对于在线多目标跟踪，我们提出了一个保守的替代方案。由于全局最优只能通过精确的非线性假设和全局赋值来实现。我们将其命名为观测中心的恢复，以信任观测值而不是估计值。

### 检测算法

`OCSORT`中检测模块基于`yolox`实现，相比于[modelzoo-yolox](../../detection/yolox/source_code/official_deploy.md)涉及到的`yolox`算法，主要区别在于预处理以及类别数，`ocsort`实现中`yolox`算法的预处理定义如下

```python
def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

### common

- focus layer
- letterbox
- Decoupled Head
- SimOTA
- Mosaic
- Mixup

## Model Info

### 模型性能

| 模型  | 源码 | MOTA | IDF1 | HOTA | dataset | input size |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: |
| ocsort_mot17_ablation |[pytorch](https://github.com/noahcao/OC_SORT/tree/master)|   	74.9  |  77.7    |   66.5    |    MOT17    |       800x1440    |
| ocsort_x_mot17 |[pytorch](https://github.com/noahcao/OC_SORT/tree/master)|   	78.0  |  77.5    |   63.2    |    MOT17    |       800x1440    |
| ocsort_x_mot20 |[pytorch](https://github.com/noahcao/OC_SORT/tree/master)|   	75.5  |  75.9    |   62.1    |    MOT20   |       896x1600    |
| ocsort_dance_model |[pytorch](https://github.com/noahcao/OC_SORT/tree/master)|   	89.4  |   54.2   |   55.1    |    dance datasets    |       800x1440    |


### 测评数据集说明

目前多目标跟踪算法中常用数据集为MOT17与MOT20，两者标签格式基本相同，MOT20数据集主要的特点是密集人群跟踪

### 评价指标说明


![image](../../../images/cv/mot/bytetrack/9440fb6f-a71b-4724-b35c-c6c53a2c3a7d.png)

![image](../../../images/cv/mot/bytetrack/90b58b99-8c60-4aef-94f1-7a525087ad42.png)

#### Classical metrics

*   _**MT**_：Mostly Tracked trajectories，成功跟踪的帧数占总帧数的80%以上的GT轨迹数量
    
*   Fragments：碎片数，成功跟踪的帧数占总帧数的80%以下的预测轨迹数量
    
*   _**ML**_：Mostly Lost trajectories，成功跟踪的帧数占总帧数的20%以下的GT轨迹数量
    
*   False trajectories：预测出来的轨迹匹配不上GT轨迹，相当于跟踪了个寂寞
    
*   ID switches：因为跟踪的每个对象都是有ID的，一个对象在整个跟踪过程中ID应该不变，但是由于跟踪算法不强大，总会出现一个对象的ID发生切换，这个指标就说明了ID切换的次数，指前一帧和后一帧中对于相同GT轨迹的预测轨迹ID发生切换，跟丢的情况不计算在ID切换中。
    

#### CLEAR MOT metrics

*   _**FP**_：总的误报数量，即整个视频中的FP数量，即对每帧的FP数量求和
    
*   _**FN**_：总的漏报数量，即整个视频中的FN数量，即对每帧的FN数量求和
    
*   _**Fragm（FM）**_：总的fragmentation数量，every time a ground truth object tracking is interrupted and later resumed is counted as a fragmentation，注意这个指标和Classical metrics中的Fragments有点不一样
    
*   _**IDSW**_：总的ID Switch数量，即整个视频中的ID Switch数量，即对每帧发生的ID Switch数量求和，这个和Classical metrics中的ID switches基本一致
    
*   _**MOTA**_：注意MOTA最大为1，由于IDSW的存在，MOTA最小可以为负无穷。
    

*   _**MOTP**_：衡量跟踪的位置误差
    

#### ID scores

![image](../../../images//cv/mot/bytetrack/5cc6bce8-2144-4f19-9476-65d14d77cd00.png)

## Build_In Deploy

- [pytorch_deploy](./source_code/pytorch_deploy.md)