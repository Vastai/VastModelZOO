
<img width="100%" src="../../../images/cv/detection/yolov12/compare.png"></a>

# Yolov12-OBB

## Code Source
```
link: https://github.com/sunsmarterjie/yolov12.git
branch: main
commit: 3a336a4adf3683d280e1a50d03fa24bbe7f24a5b
```

## Model Arch

<img width="100%" src="../../../images/cv/detection/yolov12_obb/arch.png"></a>


### pre-processing

yolov12-obb系列的预处理主要是对输入图片利用`letterbox`算子进行resize，然后进行归一化

### post-processing
yolov12-obb系列网络的后处理操作是进行box decode之后进行nms, 然后对预测的旋转边界框进行规范化处理，确保角度在有效范围内（通常为[-π/2, π/2]）；最后将将归一化的边界框坐标从模型输入尺度映射回原始图像尺度，保持检测结果的几何一致性。

### backbone
- YOLOv12的主干系统对于将原始图像数据转换为多尺度特征图至关重要，它为后续的检测任务提供了基础表征。主干网络的核心是残差高效层聚合网络（R-ELAN），它将更深的卷积层与精心设置的残差连接融合在一起。这种设计解决了梯度瓶颈问题，提高了特征重用率，增强了模型捕捉各种尺寸和形状的复杂物体细节的能力。
<img width="100%" src="../../../images/cv/detection/yolov12/R_ELAN.png"></a>

- 除了引入先进的卷积块（R-ELAN），YOLOv12还利用7×7可分离卷积等技术来减轻计算负担。这种方法有效取代了传统的大内核运算或位置编码，以更少的参数保持了空间感知能力。此外，多尺度特征金字塔可确保不同大小的物体，包括小物体或部分遮挡的物体，都能在网络中得到明显的体现。

- YOLOv12中的Neck是骨干和头部之间的通道，用于聚合和细化多尺度特征。其主要创新之一提出区域注意力模块（A2）：采用简单的划分方式将特征图划分为多个区域，避免复杂操作，在保持较大感受野的同时降低注意力计算复杂度。实验表明，使用区域注意力的 YOLOv12 - N/S/X 模型在 GPU 和 CPU 上推理速度显著提升，例如 YOLOv12 - N 在 RTX 3080 上使用 FP32 精度时，推理时间减少 0.7ms 。

<img width="100%" src="../../../images/cv/detection/yolov12/area_attention.png"></a>

### head
YOLOv12-OBB旋转目标检测头部扩展了检测头，增加了一个旋转角度预测分支，最终预测出带有旋转角度的定向边界框。
<img width="100%" src="../../../images/cv/detection/yolov12_obb/head.png"></a>

### common
- R-ELAN
- A2

## Model Info

### 测评数据集说明

![](../../../images/dataset/dota.avif)

[DOTA](https://captain-whu.github.io/DOTA/index.html) 作为一个专门的数据集，强调航空图像中的目标检测。它源自 DOTA 系列数据集，提供带注释的图像，这些图像使用旋转框检测 (OBB)捕获各种航空场景。

DOTA 展示了一个为 OBB 目标检测挑战量身定制的结构化布局：
- 图像：大量高分辨率航空图像，捕捉了不同的地形和结构。
- 定向边界框：以旋转矩形形式存在的标注，封装了目标，不受其方向的影响，非常适合捕捉飞机、船舶和建筑物等目标。

### 评价指标说明

- mAP: mean of Average Precision, 检测任务评价指标，多类别的AP的平均值；AP即平均精度，是Precision-Recall曲线下的面积
- mAP@.5: 即将IoU设为0.5时，计算每一类的所有图片的AP，然后所有类别求平均，即mAP
- mAP@.5:.95: 表示在不同IoU阈值（从0.5到0.95，步长0.05）上的平均mAP

## Build_In Deploy

### step.1 获取预训练模型

```
link: https://github.com/sunsmarterjie/yolov12.git
branch: main
commit: 3a336a4adf3683d280e1a50d03fa24bbe7f24a5b
```

- 获取原始仓库，按照此仓库的[README](https://github.com/sunsmarterjie/yolov12/tree/3a336a4adf3683d280e1a50d03fa24bbe7f24a5b?tab=readme-ov-file#installation)进行安装
- 由于yolov12官方并未提供预训练模型,因此需要自己训练模型，训练过程可以参考[Oriented Bounding Boxes Object Detection](https://docs.ultralytics.com/zh/tasks/obb/)
    - 参考[train_obb.py](./source_code/train_obb.py)，获取预训练模型
    - 使用的模型配置文件为[yolov12-obb.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/12/yolo12-obb.yaml)


### step.2 导出onnx模型
- 为适配VACC和导出onnx文件，需进行适当修改源码。
    - 目前Compiler暂不支持四维softmax算子，yolov11中DFL模块包含四维softmax算子，但是由于其后的卷积层不参与训练，因此可以将该算子后的处理截断写在host侧。综上，转换模型时可以修改[Detect](https://github.com/sunsmarterjie/yolov12/blob/3a336a4adf3683d280e1a50d03fa24bbe7f24a5b/ultralytics/nn/modules/head.py#L64)类的forward函数，替换成如下内容：
    ```python
    def forward(self, x, task_type="Detect"):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if task_type in ['Pose', 'Obb']:
            print("task_type is Pose or OBB, do not use Detect head")
            y = []
            for i in range(self.nl):
                # y.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))

                # 获取cat之前的分支结果
                cv2_output = self.cv2[i](x[i])  # 第一个分支
                cv3_output = self.cv3[i](x[i])  # 第二个分支
                # 返回未cat的各个分支
                y.append([cv2_output, cv3_output])
            return y  
        # if self.end2end:
        #     return self.forward_end2end(x)
        # for i in range(self.nl):
        #     x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        # if self.training:  # Training path
        #     return x
        # y = self._inference(x)
        # return y if self.export else (y, x)
    ```
    - obb中angle计算在build时存在问题，因此也需要放在host侧进行处理，需要修改[OBB](https://github.com/sunsmarterjie/yolov12/blob/3a336a4adf3683d280e1a50d03fa24bbe7f24a5b/ultralytics/nn/modules/head.py#L211)类的forward函数，替换成如下内容：
    ```python
    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        
        # 保存reshape之前的结果
        angle_before_reshape = [self.cv4[i](x[i]) for i in range(self.nl)]
 
        if self.export :
            x = Detect.forward(self, x, "Obb")
            # 返回reshape之前的结果
            return [x, angle_before_reshape]  # 返回未reshape的原始角度logits
        
        # # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        # angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        # if not self.training:
        #     self.angle = angle
        # x = Detect.forward(self, x)
        # if self.training:
        #     return x, angle
        # return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))
    ```
- 按原仓库安装环境
- 参考[export_onnx.py](./source_code/export_onnx.py)，导出onnx。onnx文件不包含后处理部分，输出有9个feature map。

### step.3 准备数据集
- [校准数据集](https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip)
- [评估数据集](https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip)

### step.4 模型转换
1. 根据具体模型，修改编译配置
    - [official_yolov12_obb_fp16.yaml](./build_in/build/official_yolov12_obb_fp16.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 模型编译
    ```bash
    cd yolov12_obb
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_yolov12_obb_fp16.yaml
    ```

### step.5 模型推理

- 参考[yolov12_obb_vsx.py](./build_in/vsx/python/yolov12_obb_vsx.py)生成预测的结果

    ```
    python ../build_in/vsx/python/yolov12_obb_vsx.py \
        --file_path path/to/DOTAv1/images/test \
        --model_prefix_path ./deploy_weights/ultralytics_yolov12n_obb_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-yolov12n-vdsp_params.json \
        --save_dir ./infer_output \
        --device 0

    ```

### step.6 性能测试
```bash
vamp -m ./deploy_weights/ultralytics_yolov12n_obb_fp16/mod --vdsp_params \
../build_in/vdsp_params/official-yolov12n-vdsp_params.json \
-i 1 -b 1 -d 0 -p 1
```

## Tips
- YOLOv12当前仅提供fp16的模型，后续会提供int8模型
- YOLO系列模型中，官方在精度测试和性能测试时，设定了不同的conf和iou参数
- VACC在不同测试任务中，需要分别配置build yaml内的对应参数，分别进行build模型
- `precision mode：--confidence_threshold 0.001 --nms_threshold 0.65`
- `performance mode：--confidence_threshold 0.25 --nms_threshold 0.45`

- yolov12含有Attention模块，GQA计算时，由于硬件限制seqlen需要是128的倍数；即输入分辨率需设置为128的倍数

