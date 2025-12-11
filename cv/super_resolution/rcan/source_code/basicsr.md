## BasicSR

### step.1 模型准备

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
- 将[export.py](../source_code/basicsr/export.py)移动至`BasicSR`工程目录，修改原始权重路径，导出torchscript

### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集
  - 测试高清HR图像：[DIV2K_valid_HR](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip)
  - 测试低清LR图像：[DIV2K_valid_LR_bicubi/X2](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip)

- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [basicsr_rcan.yaml](../build_in/build/basicsr_rcan.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd rcan
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/basicsr_rcan.yaml
    ```

### step.4 模型推理


    - 参考：[basicsr_vsx_inference.py](../build_in/vsx/python/basicsr_vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/basicsr_vsx_inference.py \
        --lr_image_dir  /path/to/DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --model_prefix_path deploy_weights/basicsr_rcan_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/basicsr-rcan-vdsp_params.json \
        --hr_image_dir /path/to/DIV2K/DIV2K_valid_HR \
        --save_dir ./infer_output \
        --device 0
    ```

    ```
    # int8
    mean psnr: 29.410654087264646, mean ssim: 0.8507489301523861
    
    # fp16
    mean psnr: 29.302615447487224, mean ssim: 0.85313978342054
    ```


### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[basicsr-rcan-vdsp_params.json](../build_in/vdsp_params/basicsr-rcan-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/basicsr_rcan_int8/mod \
    --vdsp_params ../build_in/vdsp_params/basicsr-rcan-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920]
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --target_path  DIV2K/DIV2K_valid_LR_bicubic/X2_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理，获得npz结果
    ```bash
    vamp -m deploy_weights/basicsr_rcan_int8/mod \
        --vdsp_params ../build_in/vdsp_params/basicsr-rcan-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,1080,1920] \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```
    
    - 解析npz结果，统计精度: [basicsr-vamp_eval.py](../build_in/vdsp_params/basicsr-vamp_eval.py)：
   ```bash
        python ../build_in/vdsp_params/basicsr-vamp_eval.py \
        --gt_dir DIV2K/DIV2K_valid_HR \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output \
        --input_shape 1080 1920 \
        --draw_dir npz_draw_result \
        --vamp_flag
   ```


## Tips
- 注意在模型输入之前有一次预处理(x/255.0)，模型forward内部又有一次预处理，所以build时的预处理和runstream推理的vdsp参数可能不太一样
