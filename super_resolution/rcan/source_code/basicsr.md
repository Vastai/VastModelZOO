## BasicSR

### step.1 获取预训练模型

```
link: https://github.com/XPixelGroup/BasicSR
branch: master
commit: 033cd6896d898fdd3dcda32e3102a792efa1b8f4
```
- 拉取原始仓库：[BasicSR](https://github.com/XPixelGroup/BasicSR)
- 内部在原始`RCAN`模型基础上进行了修改，去除了通道注意力等模块，精简计算量，以适配VASTAI板卡
- 将[arch_util.py](../source_code/basicsr/arch_util.py)，替换原始文件[basicsr/archs/arch_util.py](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/arch_util.py)，新增了`Upsample_sr4k`上采样函数
- 将[sr4k_arch.py](../source_code/basicsr/sr4k_arch.py)，移动至[basicsr/archs](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs)
- 基于[rcan_modify.yaml](../source_code/basicsr/rcan_modify.yaml)原始配置
- 将[export.py](../source_code/basicsr/export.py)移动至`BasicSR`工程目录，修改原始权重路径，导出torchscript（onnx 在opset 13时可导出，但build会报错）

### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集

- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件



### step.3 模型转换
1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[basicsr-config.yaml](../vacc_code/build/basicsr-config.yaml)，执行转换命令：
    ```bash
    vamc build ../vacc_code/build/basicsr-config.yaml
    ```
    
### step.4 benchmark

1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path DIV2K/DIV2K_valid_LR_bicubic/X2 \
    --target_path  DIV2K/DIV2K_valid_LR_bicubic/X2_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[basicsr-rcan-vdsp_params.json](../vacc_code/vdsp_params/basicsr-rcan-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/rcan-int8-max-3_1080_1920-vacc/rcan \
    --vdsp_params ../vacc_code/vdsp_params/basicsr-rcan-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/rcan-int8-max-3_1080_1920-vacc/rcan \
    --vdsp_params vacc_code/vdsp_params/basicsr-rcan-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [basicsr-vamp_eval.py](../vacc_code/vdsp_params/basicsr-vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/basicsr-vamp_eval.py \
    --gt_dir DIV2K/DIV2K_valid_HR \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/rcan \
    --input_shape 1080 1920 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


## Tips
- 在`Vaststream 1.1.7 SP11_0417`下验证
- 注意在模型输入之前有一次预处理(x/255.0)，模型forward内部又有一次预处理，所以build时的预处理和runstream推理的vdsp参数可能不太一样
