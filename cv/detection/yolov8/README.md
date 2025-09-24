
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

> Note: 数据来自官方模型性能参数

### 测评数据集说明

![](../../../images/dataset/coco.png)

[MS COCO](https://cocodataset.org/#download)的全称是Microsoft Common Objects in Context，是微软于2014年出资标注的Microsoft COCO数据集，与ImageNet竞赛一样，被视为是计算机视觉领域最受关注和最权威的比赛数据集之一。 

COCO数据集支持目标检测、关键点检测、实力分割、全景分割与图像字幕任务。在图像检测任务中，COCO数据集提供了80个类别，验证集包含5000张图片，上表的结果即在该验证集下测试。

### 评价指标说明

- mAP: mean of Average Precision, 检测任务评价指标，多类别的AP的平均值；AP即平均精度，是Precision-Recall曲线下的面积
- mAP@.5: 即将IoU设为0.5时，计算每一类的所有图片的AP，然后所有类别求平均，即mAP
- mAP@.5:.95: 表示在不同IoU阈值（从0.5到0.95，步长0.05）上的平均mAP

## Build_In Deploy

### step.1 获取预训练模型

目前Compiler暂不支持官方提供的转换脚本生成的`onnx`以及`torchscript`模型生成三件套，需基于以下脚本生成`torchscript`格式进行三件套转换(onnx暂不支持)

```python
import os
import torch
from ultralytics import YOLO

models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
size = [416, 608, 640, 1024, 2048]

for m in models:
    for s in size:
        MODEL = os.path.join('./weights', m)
        model = YOLO(MODEL)
        model.to("cpu")

        img_tensor=torch.zeros(1, 3, s, s)
        scripted_model = torch.jit.trace(model.model, img_tensor, check_trace=False).eval()

        torch.jit.save(scripted_model, os.path.join('./torchscript', m.split('.')[0] + '-' + str(s) + '.torchscript.pt'))
```

### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../common/label/coco.txt)


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [ultralytics_yolov8.yaml](./build_in/build/ultralytics_yolov8.yaml)

    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd yolov8
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/ultralytics_yolov8.yaml
    ```

### step.4 模型推理
1. runstream推理：[detection.py](../common/vsx/detection.py)
    - 配置模型路径和测试数据路径等参数

    ```
    python ../../common/vsx/detection.py \
        --file_path path/to/coco_val2017 \
        --model_prefix_path deploy_weights/ultralytics_yolov8s_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/ultralytics-yolov8s-vdsp_params.json \
        --label_txt ../../common/label/coco.txt \
        --save_dir ./runstream_output \
        --device 0
    ```

    - 精度评估，参考：[eval_map.py](../common/eval/eval_map.py)
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt ./runstream_output
    ```

    <details><summary>点击查看精度测试结果</summary>
    
    ```
    # 模型名：yolov8s-640

    # fp16
    DONE (t=2.08s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.435
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.599
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.468
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.238
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.597
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.340
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.535
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.559
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.336
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.618
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.729
    {'bbox_mAP': 0.435, 'bbox_mAP_50': 0.599, 'bbox_mAP_75': 0.468, 'bbox_mAP_s': 0.238, 'bbox_mAP_m': 0.484, 'bbox_mAP_l': 0.597, 'bbox_mAP_copypaste': '0.435 0.599 0.468 0.238 0.484 0.597'}

    # int8
    DONE (t=2.18s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.428
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.592
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.462
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.235
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.479
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.591
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.337
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.530
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.556
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.332
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.614
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.728
    {'bbox_mAP': 0.428, 'bbox_mAP_50': 0.592, 'bbox_mAP_75': 0.462, 'bbox_mAP_s': 0.235, 'bbox_mAP_m': 0.479, 'bbox_mAP_l': 0.591, 'bbox_mAP_copypaste': '0.428 0.592 0.462 0.235 0.479 0.591'}
    ```

    </details>

### step.5 性能精度测试
1. 性能测试
    - 配置[ultralytics-yolov8s-vdsp_params.json](./build_in/vdsp_params/ultralytics-yolov8s-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/ultralytics_yolov8s_run_stream_fp16/mod --vdsp_params ../build_in/vdsp_params/ultralytics-yolov8s-vdsp_params.json -i 1 p 1 -b 1 -d 0
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path path/to/coco_val2017 \
        --target_path  path/to/coco_val2017_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理获取npz结果输出
    ```bash
    vamp -m deploy_weights/ultralytics_yolov8s_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/ultralytics-yolov8s-vdsp_params.json \
        -i 1 p 1 -b 1 \
        --datalist path/to/npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz文件，参考：[npz_decode.py](../common/utils/npz_decode.py)
    ```bash
    python ../../common/utils/npz_decode.py \
        --txt result_npz --label_txt ../../common/label/coco.txt \
        --input_image_dir path/to/coco_val2017 \
        --model_size 640 640 \
        --vamp_datalist_path path/to/npz_datalist.txt \
        --vamp_output_dir npz_output
    ```

    - 精度统计，参考：[eval_map.py](../common/eval/eval_map.py)
    ```bash
    python ../../common/eval/eval_map.py \
            --gt path/to/instances_val2017.json \
            --txt path/to/result_npz
    ```

## Tips
- YOLO系列模型中，官方在精度测试和性能测试时，设定了不同的conf和iou参数
- VACC在不同测试任务中，需要分别配置build yaml内的对应参数，分别进行build模型
- `precision mode：--confidence_threshold 0.001 --nms_threshold 0.65`
- `performance mode：--confidence_threshold 0.25 --nms_threshold 0.45`