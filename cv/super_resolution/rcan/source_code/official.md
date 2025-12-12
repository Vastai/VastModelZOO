## Official

### step.1 模型准备

```
link: https://github.com/yulunzhang/RCAN
branch: master
commit: 3339ebc59519c3bb2b5719b87dd36515ec7f3ba7
```

- 拉取原始仓库：[RCAN_TestCode/code](https://github.com/yulunzhang/RCAN/tree/master/RCAN_TestCode/code)
- 内部在原始模型基础上进行了修改，去除了通道注意力等模块，精简计算量，以适配VASTAI板卡
- 原始[rcan.py](https://github.com/yulunzhang/RCAN/tree/master/RCAN_TestCode/code/model/rcan.py)，修改后[rcan_modify.py](../source_code/official/model/rcan_modify.py)
- 原始[common.py](https://github.com/yulunzhang/RCAN/tree/master/RCAN_TestCode/code/model/common.py)，修改后[common_modify.py](../source_code/official/model/common_modify.py)
- 将上述两个修改后的文件放置在原始文件同目录下
- 将[export.py](../source_code/official/export.py)移动至`RCAN_TestCode/code`目录，修改原始权重路径，导出torchscript和onnx

### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集
  - 测试高清HR图像：[DIV2K_valid_HR](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip)
  - 测试低清LR图像：[DIV2K_valid_LR_bicubi/X2](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip)

- 下载[Set5](https://github.com/yulunzhang/RCAN/tree/master/RCAN_TestCode)数据集，原仓库提供
  - 测试高清HR图像：[Set5_PNG/HR/x2](https://github.com/yulunzhang/RCAN/tree/master/RCAN_TestCode/HR/Set5/x2)
  - 测试低清LR图像：[Set5_PNG/LR/x2](https://github.com/yulunzhang/RCAN/tree/master/RCAN_TestCode/LR/LRBI/Set5/x2)

- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_rcan.yaml](../build_in/build/official_rcan.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd rcan
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_rcan.yaml
    ```

### step.4 模型推理

- 参考：[official_vsx_inference.py](../build_in/vsx/python/official_vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/official_vsx_inference.py \
        --lr_image_dir  /path/to/DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --model_prefix_path deploy_weights/official_rcan_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-rcan-vdsp_params.json \
        --hr_image_dir /path/to/DIV2K/DIV2K_valid_HR \
        --save_dir ./infer_output \
        --device 0
    ```

    ```
    # int8
    mean psnr: 32.3262838692537, mean ssim: 0.8812454553357942
    
    # fp16
    mean psnr: 32.28759321073234, mean ssim: 0.8890894206115039
    ```


### step.5 性能测试
1. 性能测试
    - 配置vdsp参数[official-rcan-vdsp_params.json](../build_in/vdsp_params/official-rcan-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_rcan_int8/mod \
    --vdsp_params ../build_in/vdsp_params/official-rcan-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920]
    ```
2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；
    - 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --target_path  DIV2K/DIV2K_valid_LR_bicubic/X2_npz \
        --text_path npz_datalist.txt
    ```

    - 精度测试，vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/official_rcan_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-rcan-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,1080,1920] \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 参考：[official-vamp_eval.py](../build_in/vdsp_params/official-vamp_eval.py)，解析npz结果，统计精度：
    ```bash
    python ../build_in/vdsp_params/official-vamp_eval.py \
        --gt_dir DIV2K/DIV2K_valid_HR \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output \
        --input_shape 1080 1920 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```

## Tips
- 修改版的RCAN内部定义为：SR4K


