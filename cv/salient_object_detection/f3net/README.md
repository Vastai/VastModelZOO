
# F3Net

[F3Net: Fusion, Feedback and Focus for Salient Object Detection](https://arxiv.org/abs/1911.11445)

## Code Source
```
link: https://github.com/weijun88/F3Net
branch: master
commit: eecace3adf1e8946b571a4f4397681252f9dc1b8
```

## Model Arch

### pre-processing

F3Net系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至一定尺寸，然后对其进行归一化等操作：

```python
image = cv2.imread(image_file)
img = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mean = np.array([124.55, 118.90, 102.94])
std  = np.array([ 56.77,  55.97,  57.50])
img = (img - mean) / std
```

### post-processing

F3Net系列网络的后处理操作，主要有sigmoid和反归一化：
```python
out = np.squeeze(heatmap)
out = torch.from_numpy(out)
pred = (torch.sigmoid(out) * 255).cpu().numpy()
```

### backbone

大多数现有的显著目标检测模型都通过聚合从卷积神经网络中提取的多级特征取得了长足的进步。然而，由于不同卷积层的感受野不同，这些层生成的特征之间存在很大差异。常见的特征融合策略（加法或连接）忽略了这些差异，并可能导致次优解决方案。
F3Net通过以下优化来解决上述问题：
- 我们引入交叉特征模块(CFM)来融合不同层次的特征，能够提取特征之间的共享部分，抑制彼此的背景噪声，补充彼此缺失的部分。
- 我们提出了用于SOD的级联反馈解码器(CFD)，它可以将高分辨率和高语义的特征反馈到之前的特征上，从而对其进行纠正和改进，以更好地生成显著图。
- 我们设计像素位置感知损失，为不同的位置分配不同的权重。它可以更好地挖掘特征中包含的结构信息，帮助网络更多地关注细节区域。

<div  align="center">
<img src="../../../images/cv/salient_object_detection/f3net/f3net.png" width="80%" height="80%">
</div>

### common

- Cross Feature Module
- Cascaded Feedback Decoder

## Model Info

### 模型性能
| Models  | Flops(G) | Params(M) | MAE ↓ | avg F-Measure ↑ | SM ↑ | Shapes |
| :---: | :--: | :--: | :---: | :--------: | :---: | :--------: |
| [F3Net](https://github.com/weijun88/F3Net) | 77.240  |  25.537  | 0.033  |  0.924  | 0.925  | 3x512x512  |
| F3Net **vacc fp16** |  -  |  -  |  0.048  |  0.898  | 0.899  | 3x512x512  |
| F3Net **vacc kl_divergence int8** |  -  |  -  |   0.046  |  0.904  | 0.902  |  3x512x512  |


### 测评数据集说明


[ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)数据集，由香港中文大学的Yan等人于2013年建立, 包含了1000张图像, 这些图像由互联网得到。该数据集中的显著物体包含较复杂的结构, 且背景具备一定的复杂性。


<div  align="center">
<img src="../../../images/dataset/ecssd.jpg" width="80%" height="70%">
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


## Build_In Deploy

### step.1 获取预训练模型
- 观察[test.py#L56](https://github.com/weijun88/F3Net/blob/master/src/test.py#L56)，模型forward后只使用到第二个返回值。为减少模型推理时的数据拷贝，修改[src/net.py#L186](https://github.com/weijun88/F3Net/blob/master/src/net.py#L186)，只返回第二值，`return pred2`
- 在原仓库[test.py#L33](https://github.com/weijun88/F3Net/blob/master/src/test.py#L33)，定义模型和加载训练权重后，添加以下脚本，执行即可导出onnx和torchscript：
```python
input_shape = (1, 3, 512, 512)
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)
with torch.no_grad():
    scripted_model = torch.jit.trace(self.net, input_data).eval()
    scripted_model.save(self.snapshot.replace(".pth", ".torchscript.pt"))
    scripted_model = torch.jit.load(self.snapshot.replace(".pth", ".torchscript.pt"))

    torch.onnx.export(self.net, input_data, self.snapshot.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
    dynamic_axes= {
        "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
        "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}
        }
    )
```

### step.2 准备数据集
- 下载[ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)数据集

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_f3net.yaml](./build_in/build/official_f3net.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd dhsnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_f3net.yaml
    ```

### step.4 模型推理
1. runstream
    - 参考：[official_vsx_inference.py](./build_in/vsx/python/official_vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/official_vsx_inference.py \
        --image_dir /path/to/sod/ECSSD/image \
        --model_prefix_path deploy_weights/official_f3net_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-f3net-vdsp_params.json \
        --save_dir ./runstream_output \
        --device 0
    ```

    - 统计精度信息, 基于[eval.py](../common/eval/eval.py)
        ```
        python ../../common/eval/eval.py --dataset-json path/to/config_dataset.json --method-json path/to/source_code/config_method.json
        ```
        - 来自[PySODEvalToolkit](https://github.com/lartpang/PySODEvalToolkit)工具箱
        - 配置数据集路径：[config_dataset.json](../common/eval/examples/config_dataset.json)
        - 配置模型推理结果路径及图片格式：[config_method.json](../common/eval/examples/config_method.json)
    
    <details><summary>点击查看精度统计结果</summary>

    - fp16精度
    Dataset: ECSSD

    | methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
    |-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
    | Method1   | 0.099 |          0.79 |         0.783 |         0.788 |          0.859 |          0.814 |           1 |       0.793 |   0.857 |   0.842 |   0.852 | 0.806 | 0.754 |

    - int8精度
    Dataset: ECSSD

    | methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
    |-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
    | Method1   | 0.075 |          0.84 |         0.831 |         0.836 |          0.916 |          0.869 |           1 |       0.812 |   0.889 |   0.878 |   0.885 | 0.845 | 0.801 |

    </details>

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[official-f3net-vdsp_params.json](./build_in/vdsp_params/official-f3net-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_f3net_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/official-f3net-vdsp_params.json \
        -i 1 p 1 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path sod/ECSSD/image \
        --target_path sod/ECSSD/image_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果
    ```bash
    vamp -m deploy_weights/official_f3net_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/official-f3net-vdsp_params.json \
        -i 1 p 1 -b 1 \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果，基于[vamp_eval.py](./build_in/vdsp_params/vamp_eval.py)
    ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
        --src_dir data/ECSSD/image \
        --gt_dir data/ECSSD/mask \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output \
        --input_shape 512 512 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```

    - 统计精度信息, 基于[eval.py](../common/eval/eval.py)
        ```
        python ../../common/eval/eval.py --dataset-json path/to/config_dataset.json --method-json path/to/source_code/config_method.json
        ```
        - 来自[PySODEvalToolkit](https://github.com/lartpang/PySODEvalToolkit)工具箱
        - 配置数据集路径：[config_dataset.json](../common/eval/examples/config_dataset.json)
        - 配置模型推理结果路径及图片格式：[config_method.json](../common/eval/examples/config_method.json)
    

