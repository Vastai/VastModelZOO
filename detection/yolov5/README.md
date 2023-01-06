
<div align=center><img src="../../images/yolov5/info.png"></div>

# YoloV5

[YOLOv5 github](https://github.com/ultralytics/yolov5/tree/v6.1)

## Model Arch

### pre-processing

yolov5系列的预处理主要是对输入图片利用`letterbox`算子进行resize，然后进行归一化

### post-processing

yolov5系列的后处理操作是利用anchor以及网络预测特征图进行box decode，然后进行nms操作

### backbone

与YOLOv4相比，YOLOv5在Backbone部分变化不大，采用了CSP结构以及SPPF结构。YOLOv5在v6.0版本后相比之前版本有一个很小的改动，即把网络的Focus模块换成了6x6大小的卷积层。两者在理论上其实等价的，但是对于现有的一些GPU设备（以及相应的优化算法）使用6x6大小的卷积层比使用Focus模块更加高效。

### neck

Neck部分将SPP模块换成了SPPF模块，两者的作用是一样的，但后者效率更高。SPP结构是将输入并行通过多个不同大小的MaxPool，然后做进一步融合，能在一定程度上解决目标多尺度问题。而SPPF结构是将输入串行通过多个5x5大小的MaxPool层。通过对比可以发现，两者的计算结果是一模一样的，但SPPF比SPP计算速度快了不止两倍。

### head

yolov5网络head层利用FPN+PAN的结构，融合不同维度的特征，最后有三个输出头，分别对应8、16、32的stride，不同stride的输出预测不同尺寸的目标;

训练时，yolov5在网格上生成相应的anchor框和其对应的cls以及conf。box loss采用了CIOU的方式来进行回归，conf以及cls loss使用BCEWithLogitsLoss

### common

- SPPF
- letterbox
- CIoU Loss

## Model Info

### 模型性能

**Note:** 下表中的精度在conf为0.25、nms_thresh为0.45的参数设置下统计的结果

| model name |                         codebase                          | mAP@.5 | mAP@.5:.95 | flops(G) | params(M) | input size |
| :--------: | :-------------------------------------------------------: | :----: | :--------: | :------: | :-------: | :--------: |
|  yolov5n   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  36.1  |    23.1    |   4.5    |    1.9    |    640     |
|  yolov5s   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  48.4  |    32.9    |   16.5   |    7.2    |    640     |
|  yolov5m   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  56.3  |    40.7    |   49.0   |   21.2    |    640     |
|  yolov5l   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  60.0  |    44.5    |  109.1   |   46.5    |    640     |
|  yolov5x   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  61.8  |    46.3    |  205.7   |   86.7    |    640     |
|  yolov5n6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  37.2  |    25.4    |   4.6    |    3.2    |    640     |
|  yolov5s6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  48.9  |    34.2    |   16.8   |   12.6    |    640     |
|  yolov5m6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  56.5  |    41.8    |   50.0   |   35.7    |    640     |
|  yolov5l6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  59.7  |    44.7    |  111.4   |   76.8    |    640     |
|  yolov5x6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  61.3  |    46.3    |  209.8   |   140.7   |    640     |
|  yolov5n   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  33.5  |    21.4    |   1.9    |    1.9    |    416     |
|  yolov5s   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  44.7  |    30.3    |   7.0    |    7.2    |    416     |
|  yolov5m   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  51.8  |    37.2    |   20.7   |   21.2    |    416     |
|  yolov5l   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  55.7  |    41.0    |   46.1   |   46.5    |    416     |
|  yolov5x   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  58.1  |    43.0    |   86.9   |   86.7    |    416     |
|  yolov5n6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  30.2  |    20.1    |   2.2    |    3.2    |    448     |
|  yolov5s6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  40.8  |    28.1    |   8.2    |   12.6    |    448     |
|  yolov5m6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  50.1  |    36.5    |   24.5   |   35.7    |    448     |
|  yolov5l6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  53.5  |    39.5    |   54.5   |   76.8    |    448     |
|  yolov5x6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  55.4  |    41.1    |  102.7   |   140.7   |    448     |
|  yolov5n   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  29.0  |    18.5    |   1.1    |    1.9    |    320     |
|  yolov5s   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  40.0  |    26.8    |   4.1    |    7.2    |    320     |
|  yolov5m   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  47.9  |    33.9    |   12.2   |   21.2    |    320     |
|  yolov5l   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  51.5  |    37.5    |   27.3   |   46.5    |    320     |
|  yolov5x   | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  53.9  |    39.4    |   51.4   |   86.7    |    320     |
|  yolov5n6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  22.1  |    14.4    |   1.1    |    3.2    |    320     |
|  yolov5s6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  31.9  |    21.7    |   4.2    |   12.6    |    320     |
|  yolov5m6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  42.7  |    30.6    |   12.5   |   35.7    |    320     |
|  yolov5l6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  46.4  |    33.7    |   27.8   |   76.8    |    320     |
|  yolov5x6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  48.3  |    35.2    |   52.4   |   140.7   |    320     |
|  yolov5n6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  41.8  |    28.9    |   9.0    |    3.2    |    896     |
|  yolov5s6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  53.1  |    38.0    |   33.0   |   12.6    |    896     |
|  yolov5m6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  59.7  |    44.9    |   98.0   |   35.7    |    896     |
|  yolov5l6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  62.6  |    47.5    |  218.3   |   76.8    |    896     |
|  yolov5x6  | [yolov5](https://github.com/ultralytics/yolov5/tree/v6.1) |  63.7  |    48.5    |  411.2   |   140.7   |    896     |

### 测评数据集说明

<div align=center><img src="../../images/datasets/coco.png"></div>

yolov5系列算法在[MS COCO](https://cocodataset.org/#download)数据集下测试，MS COCO的全称是Microsoft Common Objects in Context，是微软于2014年出资标注的Microsoft COCO数据集，与ImageNet竞赛一样，被视为是计算机视觉领域最受关注和最权威的比赛数据集之一。

COCO数据集支持目标检测、关键点检测、实力分割、全景分割与图像字幕任务。在图像检测任务中，COCO数据集提供了80个类别，验证集包含5000张图片，上表的结果即在该验证集下测试。

### 评价指标说明

- mAP: mean of Average Precision, 检测任务评价指标，多类别的AP的平均值；AP即平均精度，是Precision-Recall曲线下的面积
- mAP@.5: 即将IoU设为0.5时，计算每一类的所有图片的AP，然后所有类别求平均，即mAP
- mAP@.5:.95: 表示在不同IoU阈值（从0.5到0.95，步长0.05）上的平均mAP

## VACC

### step.1 获取预训练模型

```bash
git clone -b v6.1 git@github.com:ultralytics/yolov5.git
cd yolov5
python export.py --weights /path/to/weights_path --include torchscript onnx --imgsz $img_shape
```
- weights: 原模型的路径
- imgsz: 模型输入尺寸
- include: 导出模型格式，目前仅支持onnx与torchscript格式转换

### step.2 准备数据集
该模型使用coco2017数据集，请到coco官网自行下载coco2017，针对`int8`校准数据可从该数据集中任选50张作为校准数据集，[coco2017](https://cocodataset.org/#download)

```
├── COCO
|   ├── val
|   |    ├── 000000000139.jpg
│   |    ├── 000000000285.jpg
│   |    ├── ......
|   ├── instances_val2017.json
```

```bash
# label.txt
person
bicycle
car
motorcycle
airplane
bus
train
```

### step.3 模型转换
1. 根据具体模型修改配置文件, [config](./build_config/torch_yolov5.yaml)
2. 命令行执行转换
   ```bash
   vamc build ./build_config/torch_yolov5.yaml
   ```

### step.4 模型推理
1. 根据step.3配置模型三件套信息，[model_info](./model_info/model_info_yolov5.json)
2. 配置数据预处理流程，[vdsp_params](./model_info/vdsp_params_yolov5_letterbox_rgb.json)
3. 执行推理，调用入口[sample_det](../../inference/detection/sample_det.py)，源码可参考[image_detection](../../inference/detection/image_detection.py)
    ```bash
    # 执行精度评测
    cd ../../inference/detection
    python sample_det.py --file_path /path/to/datasets/coco_val2017/ --label_txt ../data/label/coco.txt --save_dir ../output --model_info ./model_info/model_info_yolov5.json --vdsp_params_info ./model_info/vdsp_params_yolov5_letterbox_rgb.json
    ```
   - file_path: 待测试数据，可以是单张图片或文件夹，测试coco数据集可以直接指定coco数据集的路径
   - model_info: 指定上阶段转换的模型路径
   - vdsp_params_info: 指定测试图片的size以及均值方差等数值，图片size要和转换模型时的size一致，否则结果会有异常
   - label_txt: 模型的标签设置，参考step.2
   - save_dir: 测试结果保存路径，会存储图片结果以及txt结果(可用于精度测试)

### step.5 评估
1. 执行完step.4， 根据生成结果txt，执行精度评估，调用入口[coco_map](../../inference/detection/coco_map.py)
   ```bash
   python coco_map.py --gt /path/to/instances_val2017.json --txt output
   ```
   - gt: 表示coco数据集ground-truth
   - txt: step.4阶段测试的结果

2. 基于`VE1`性能参考
   > nms threshold: 0.45, confidence: 0.25

   | model name | data type | through output | latency | batchsize | quant mode | mAP@0.5-0.95 |  shape  |
   | :--------: | :-------: | :------------: | :-----: | :-------: | :--------: | :----------: | :-----: |
   |  yolov5n   |   FP16    |      376       |  2.72   |     2     |     \      |    23.22     | 640*640 |
   |  yolov5n   |   INT8    |      972       |  1.18   |     2     |    max     |    22.08     | 640*640 |
   |  yolov5s   |   FP16    |      494       |  2.02   |     1     |     \      |    32.92     | 640*640 |
   |  yolov5s   |   INT8    |      986       |  1.18   |     2     |    max     |    31.07     | 640*640 |
   |  yolov5m   |   FP16    |      225       |  4.44   |     1     |     \      |    40.50     | 640*640 |
   |  yolov5m   |   INT8    |      728       |  1.46   |     2     | percentile |    38.57     | 640*640 |
   |  yolov5l   |   FP16    |      118       |  8.47   |     1     |     \      |    44.54     | 640*640 |
   |  yolov5l   |   INT8    |      434       |  2.35   |     2     | percentile |    43.62     | 640*640 |
   |  yolov5x   |   FP16    |       48       |  20.83  |     1     |     \      |    46.25     | 640*640 |
   |  yolov5x   |   INT8    |      232       |  4.31   |     1     | percentile |    45.00     | 640*640 |
   |  yolov5n6  |   FP16    |      502       |  2.16   |     2     |     \      |    25.14     | 640*640 |
   |  yolov5n6  |   INT8    |      948       |  1.22   |     2     | percentile |    23.78     | 640*640 |
   |  yolov5s6  |   FP16    |      487       |  2.05   |     1     |     \      |    33.94     | 640*640 |
   |  yolov5s6  |   INT8    |      943       |  1.14   |     2     | percentile |    32.66     | 640*640 |
   |  yolov5m6  |   FP16    |      209       |  4.78   |     1     |     \      |    41.968    | 640*640 |
   |  yolov5m6  |   INT8    |      656       |  1.61   |     2     |    max     |    40.00     | 640*640 |
   |  yolov5l6  |   FP16    |      105       |  9.52   |     1     |     \      |    44.860    | 640*640 |
   |  yolov5l6  |   INT8    |      391       |  2.77   |     2     |    max     |    42.66     | 640*640 |
   |  yolov5x6  |   FP16    |       45       |  22.22  |     1     |     \      |    46.41     | 640*640 |
   |  yolov5x6  |   INT8    |      189       |  5.29   |     1     | percentile |    44.24     | 640*640 |