
<img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png"></a>

# YoloV8

## Code Source
```
link: https://github.com/ultralytics/ultralytics
branch: main
commit: e21428ca4e0ed7f1d8fd8abad728989644b6a300
```

## Model Arch

### pre-processing

yolov8系列的预处理主要是对输入图片利用`letterbox`算子进行resize，然后进行归一化

### post-processing

yolov8系列的后处理操作相比于yolov5没有改动，即进行box decode之后进行nms即可

### backbone

Yolov8骨干网络和Neck部分参考了YOLOv7 ELAN设计思想，将YOLOv5的C3结构换成了梯度流更丰富的C2f结构，并对不同尺度模型调整了不同的通道数，属于对模型结构精心微调，不再是无脑一套参数应用所有模型，大幅提升了模型性能

### head

yolov8系列Head部分相比YOLOv5改动较大，换成了目前主流的解耦头结构，将分类和检测头分离，同时也从Anchor-Based换成了Anchor-Free，Loss计算方面采用了TaskAlignedAssigner正样本分配策略，并引入了 Distribution Focal Loss

### common

- C2f
- SPPF
- letterbox
- DFL

## Model Info

### 模型性能

| 模型  | 源码 | mAP@.5 | mAP@.5:.95 | flops(B) | params(M) | input size |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: |
| yolov8n |[ultralytics](https://github.com/ultralytics/ultralytics)|   -   |   37.3   |   8.7    |    3.2    |        640    |
| yolov8s |[ultralytics](https://github.com/ultralytics/ultralytics)|   -   |   44.9   |   28.6    |    11.2   |        640    |
| yolov8m |[ultralytics](https://github.com/ultralytics/ultralytics)|   -   |   50.2   |   78.9    |    25.9    |        640    |
| yolov8l |[ultralytics](https://github.com/ultralytics/ultralytics)|   -   |   52.9   |   165.2    |    43.7    |        640    |
| yolov8x |[ultralytics](https://github.com/ultralytics/ultralytics)|   -   |   53.9   |  257.8   |    68.2    |        640    |

### 测评数据集说明

![](../../images/dataset/coco.png)

[MS COCO](https://cocodataset.org/#download)的全称是Microsoft Common Objects in Context，是微软于2014年出资标注的Microsoft COCO数据集，与ImageNet竞赛一样，被视为是计算机视觉领域最受关注和最权威的比赛数据集之一。 

COCO数据集支持目标检测、关键点检测、实力分割、全景分割与图像字幕任务。在图像检测任务中，COCO数据集提供了80个类别，验证集包含5000张图片，上表的结果即在该验证集下测试。

### 评价指标说明

- mAP: mean of Average Precision, 检测任务评价指标，多类别的AP的平均值；AP即平均精度，是Precision-Recall曲线下的面积
- mAP@.5: 即将IoU设为0.5时，计算每一类的所有图片的AP，然后所有类别求平均，即mAP
- mAP@.5:.95: 表示在不同IoU阈值（从0.5到0.95，步长0.05）上的平均mAP

## VACC部署

- [ultralytics_deploy](./source_code/ultralytics_deploy.md)