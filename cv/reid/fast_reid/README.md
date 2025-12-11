# Fast_Reid

[code](https://github.com/JDAI-CV/fast-reid)

## Model Arch

FastReID是一个SOTA级的ReID方法集合工具箱（SOTA ReID Methods and Toolbox），同时面向学术界和工业界落地，此外该团队还发布了在多个不同任务、多种数据集上的SOTA模型。
FastReID中实现的四大ReID任务：
- 行人重识别
- 部分区域的行人重识别
- 跨域的行人重识别
- 车辆的重识别

FastReID提供了针对ReID任务的完整的工具箱，包括训练、评估、微调和模型部署，另外实现了在多个任务中的最先进的模型。

1）模块化和灵活的设计，方便研究者快速将新的模块插入和替换现有流程中，帮助学界快速验证新思路；

2）易于管理的系统配置，FastReID用PyTorch实现，可提供在多GPU服务器的快速训练，模型结构、训练和测试可以用YAML文件方便定义，并对每一块提供了众多可定义选项（主干网、训练策略、聚合策略、损失函数等）。

3）丰富的评估系统，不仅实现了CMC评估指标，还包括ROC、mINP等，可以更好的反应模型性能。

4）易于工程部署，FastReID不仅提供了有效的知识蒸馏模块以得到精确且高效的轻量级模型，而且提供了PyTorch->Caffe和PyTorch->TensorRT模型转换工具。

5）众多的State-of-the-art预训练模型，官方计划发布人员重识别（person re-id）,部分可见的人员重识别（ partial re-id）, 跨域人员重识别（cross-domain re-id） 和 车辆重识别（vehicle re-id） 等众多模型。


### pre-processing

FastReID工具的预处理Pre-processing模块就是各种数据增广方法，如Resize、Flipping、Random erasing、Auto-augment、Random patch、Cutout等

### post-processing

FastReID工具的后处理部分，指对检索结果的处理，包括K-reciprocal coding 和 Query Expansion (QE) 两种重排序方法。

### backbone

骨干网（Backbone），包括主干网的选择（如ResNet,ResNest,ResNeXt等）和可以增强主干网表达能力的特殊模块（如non-local、instance batch normalization (IBN)模块等）

聚合模块（Aggregation），用于将骨干网生成的特征聚合成一个全局特征，如max pooling, average pooling, GeM pooling ， attention pooling等方法；

### head

Head 模块，用于对生成的全局特征进行归一化、纬度约减等。

### common

- residual layer
- non-local
- instance batch normalization
- max pooling
- average pooling
- attention pooling

## Model Info

### 模型性能

| 模型  | 源码 | Rank@1 | mAP | mINP | dataset |
| :---: | :--: | :--: | :--: | :---: | :---: | 
| market_bot_R50 |[official](https://github.com/JDAI-CV/fast-reid)|   94.4%	  |   86.1%  |   59.4%     | Market1501 |
| market_bot_S50 |[official](https://github.com/JDAI-CV/fast-reid)|   95.2%   |   88.7%	 |   66.9%     | Market1501 |
| market_bot_R50_ibn |[official](https://github.com/JDAI-CV/fast-reid)|   94.9%  |   87.6% |   64.1%  | Market1501 |
| market_bot_R101_ibn |[official](https://github.com/JDAI-CV/fast-reid)|   95.4%   |   88.9%  |   67.4%| Market1501 |

### 测评数据集说明

<div align=center><img src="../../../images/dataset/Market1501.jpg"></div>

Market-1501 数据集在清华大学校园中采集，夏天拍摄，在 2015 年构建并公开。它包括由6个摄像头（其中5个高清摄像头和1个低清摄像头）拍摄到的 1501 个行人、32668 个检测到的行人矩形框。每个行人至少由2个摄像头捕获到，并且在一个摄像头中可能具有多张图像。训练集有 751 人，包含 12,936 张图像，平均每个人有 17.2 张训练数据；测试集有 750 人，包含 19,732 张图像，平均每个人有 26.3 张测试数据。3368 张查询图像的行人检测矩形框是人工绘制的，而 gallery 中的行人检测矩形框则是使用DPM检测器检测得到的。该数据集提供的固定数量的训练集和测试集均可以在single-shot或multi-shot测试设置下使用。


### 评价指标说明

- Rank1(CMC，Cumulative Matching Characteristics)
    ```
    Rank1是我们在阅读ReID相关论文中最常见的两个指标之一，它的计算如下：
    1）首先定义一个指示函数表示q，i两张图片是否具有相同标签：
    2）那么计算rank1时，只需统计所有查询图片与他们的第一个返回结果是否相同，Q为全体查询图片query的集合， 为q这张查询图片对应的图像库中第 i个返回结果的标签：

    Rank1可以表示图像的第一检索目标的准确率，同样的计算方式也可以获得Rank5，Rank10等指标。
    ```
- mAP
    ```
    在ReID中MAP表示所有检索结果的准确率，是常用的两个ReID指标之一。计算过程如下：
    1）P：精度，即对某一张probe图片，计算前k个返回结果中与查询图片相同ID的数量比例。
    2）AP@n：平均精度，即在前n个返回结果中，只对那些返回结果正确的位置的精度进行平均，即nq为q这张查询图片在前k个返回结果中有多少个正确返回结果。
    3）mAP@n：对所有probe图片，均计算其AP，将这些结果求均值。

    ```
- mINP
    ```
    mINP是在2020年初发表的《Deep Learning for Person Re-identifification: A Survey and Outlook》中提到的一个指标。
    该指标是为了评价一个模型搜索到最难找到样本的能力。
    大致意思是现在的指标只是考虑在召回的前k中去做评价，这样是不准确的，有一些正确样本排序排在很后但是已有的评价指标没有去考虑到它们，为此作者觉得需要加入个惩罚项（NP，negative penalty）来考虑该问题。

    ```

## Build_In Deploy

### step.1 模型准备

原始仓库 [fast-reid](https://github.com/JDAI-CV/fast-reid/tree/master) 
```bash
git clone https://github.com/JDAI-CV/fast-reid.git
```

代码中提供了模型转onnx的脚本，可以用以下命令将原始模型转为onnx格式

```bash
python tools/deploy/onnx_export.py --config-file configs/Market1501/bagtricks_R50.yml --name market_bot_R50 --output models --opts MODEL.WEIGHTS models/market_bot_R50.pth MODEL.DEVICE "cpu"
```

**若在CPU侧执行后续流程时，则需注释`fast-reid/fastreid/data/data_utils.py`中`cuda`相关代码行。source_code/data_utils.py为去掉cuda实现的代码，可以直接替换同名文件**


### step.2 准备数据集
- 下载[Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)数据集并进行解压到当前workspace/datasets文件夹
- 解压后的路径格式为：
```
datasets
├── Market-1501-v15.09.15
└── README.md
```

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_fast_reid.yaml](./build_in/build/official_fast_reid.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd reid
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_fast_reid.yaml
    ```

### step.4 模型推理
1. 需要注意运行该推理需要依赖fast-reid项目，因此需要将fast-reid项目放置source_code目录下

2. runstream
    - 参考[reid_vsx.py](./build_in/vsx/python/reid_vsx.py)生成预测的txt结果

    ```
    python ../build_in/vsx/python/reid_vsx.py \
        --model_prefix_path deploy_weights/official_fast_reid_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-market_bot_R50_ibn-vdsp_params.json \
        --config-file ../source_code/fast-reid/configs/Market1501/bagtricks_R50.yml \
        --device 0
    ```

    ```
    # fp16
    OrderedDict([('Rank-1', 90.58788418769836), ('Rank-5', 96.16983532905579), ('Rank-10', 97.74346947669983), ('mAP', 75.01060962677002), ('mINP', 40.51458537578583), ('metric', 82.79924392700195)])

    ```

### step.5 性能精度测试
1. 性能测试
    ```bash
    vamp -m deploy_weights/official_fast_reid_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/official-market_bot_R50_ibn-vdsp_params.json \
        -i 1 p 1 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，参考：[image2npz.py](../common/image2npz.py)
    ```bash
    python ../../common/image2npz.py --target_path data_npz --text_path datalist.txt
    ```

    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/official_fast_reid_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/official-market_bot_R50_ibn-vdsp_params.json \
        --datalist datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果，参考：[npz_decode.py](./build_in/vdsp_params/npz_decode.py)
    ```bash
    python ../build_in/vdsp_params/npz_decode.py --config-file ../source_code/fast-reid/configs/Market1501/bagtricks_R50.yml
    ```
