
# DINOv2

[DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

## Code Source
```
link: https://github.com/facebookresearch/dinov2
branch: main
commit: e1277af2ba9496fbadf7aec6eba56e8d882d1e35
```

## Model Arch

DINOv2(Dual-Stage Implicit Object-Oriented Network) 是一种双阶段训练的Transformer模型，它以其强大的图像特征提取能力和广泛的适用性，无需微调即可在多种下游任务(图像分类、图像分割、图像检索等)中表现出色。


### pre-processing

DINOv2系列网络的预处理操作，可以按照如下步骤进行：

```python
  from torchvision import transforms as pth_transforms
  transform = pth_transforms.Compose([
      pth_transforms.Resize([224, 224]),
      pth_transforms.ToTensor(),
      pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  ])
```

### post-processing
DINOv2系列网络的模型输出为特征图，不同任务不同后处理。


### backbone

DINOv2模型是一个VisionTransformer网络，通过patch_size切分图像进行embedding，进入transformer block。

### common

- vit


### 测评数据集说明


Roxford5k数据集是一个用于图像检索任务的数据集，通常用于评估图像检索算法的性能。它包含5000张图像，这些图像来自于不同的场景和物体。数据集的特点包括：


Roxford5k数据集包含通过搜索Flickr从特定牛津地标收集的5062张图像。该集合已被手动注释，以生成11个不同地标的综合地面实况，每个地标由5个可能的查询表示。
- 图像类别：数据集中包括多个类别的图像，例如建筑、风景、街道等。
- 标注信息：每张图像都有相关的标签，通常包括位置、物体类别等信息。这些标签用于训练和评估图像检索算法的准确性。
- 应用场景：Roxford5k常用于基准测试和算法评估，尤其是在深度学习和计算机视觉领域，研究者们用它来测试图像检索和匹配算法的效果。
- 挑战性：由于图像具有复杂的背景和多样的视角，这为检索算法提出了挑战，使其成为一个受欢迎的测试数据集。


### 指标说明
- 平均精度（Mean Average Precision, MAP）：对于多个查询，计算每个查询的平均精度，然后取平均值，常用于评估整体检索系统的性能
- 准确率@K（Precision@K）：检索结果中前K个结果的准确率，常用于评估系统在前N个结果中的表现

## Build_In Deploy

### step.1 获取预训练模型
- 参考[convert-to-onnx.py](./source_code/convert-to-onnx.py)导出onnx

### step.2 准备数据集
- 参考[download_dataset.py](./source_code/download_dataset.py)，下载roxford5k数据集

### step.3 模型转换
- 需要注意当前只支持FP16的模型。
1. 根据具体模型，修改编译配置
    - [official_dinov2.yaml](./build_in/build/official_dinov2.yaml)
        
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 模型编译
    ```bash
    cd dinov2
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_dinov2.yaml
    ```

### step.4 模型推理

- 参考: [official_vsx_inference.py](./build_in/vsx/python/official_vsx_inference.py)
    - 获取对应的[elf文件](../../classification/common/elf/)
    ```bash
    python ../build_in/vsx/python/official_vsx_inference.py \
        --dataset_root /path/to/roxford5k/jpg \
        --model_prefix deploy_weights/official_dinov2_fp16/mod \
        --norm_elf_file /path/to/elf/normalize \
        --space_to_depth_elf_file /path/to/elf/space_to_depth \
        --dataset_conf /path/to/roxford5k/gnd_roxford5k.pkl \
        --device 0
    ```
- 精度结果在打印信息最后，如下：
    ```
    mAP M: 79.6, H: 58.18
    mP@k[ 1  5 10] M: [98.57 94.52 91.38], H: [92.86 80.05 70.05]
    ```

### step.5 性能测试
1. 性能测试
    - 配置vdsp参数[official-dinov2_vitl14_reg4-vdsp_params.json](./build_in/vdsp_params/official-dinov2_vitl14_reg4-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_dinov2_fp16/mod \
    --vdsp_params ../build_in/vdsp_params/official-dinov2_vitl14_reg4-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,224,224]
    ```
