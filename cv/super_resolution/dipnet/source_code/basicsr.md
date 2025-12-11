## BasicSR

### step.1 获取预训练模型

```
link: https://github.com/xiumu00/DIPNet
branch: master
commit: 69f43a7873a866a492ee316bc579e56b6f861170
```

- 训练：
  - 拉取[BasicSR](https://github.com/XPixelGroup/BasicSR)仓库
  - 将本仓库[dipnet_arch.py](../source_code/basicsr/archs/dipnet_arch.py)移动至[basicsr/archs](https://github.com/XPixelGroup/BasicSR/tree/master/basicsr/archs)目录下
  - 训练配置[train_DIPNet_x2_m4c16_prelu_RGB_noSEmini.yml](basicsr/archs/train_DIPNet_x2_m4c16_prelu_RGB_noSEmini.yml)，参考BasicSR训练进行流程
- 导出
  - 参考[dipnet_arch.py](../source_code/basicsr/archs/dipnet_arch.py)，修改原始权重路径，导出torchscript和onnx


### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集

- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [basicsr_dipnet.yaml](../build_in/build/basicsr_dipnet.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd dipnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/basicsr_dipnet.yaml
    ```

### step.4 模型推理

    - 参考[basicsr_vsx_inference.py](../build_in/vsx/python/basicsr_vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/basicsr_vsx_inference.py \
        --lr_image_dir  /path/to/DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --model_prefix_path deploy_weights/basicsr_dipnet_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/basicsr-dipnet-vdsp_params.json \
        --hr_image_dir /path/to/DIV2K/DIV2K_valid_HR \
        --save_dir ./infer_output \
        --device 0
    ```

    ```
    # fp16
    mean psnr: 32.88215295762826, mean ssim: 0.9057846278816606

    # int8 
    mean psnr: 31.75613463739251, mean ssim: 0.8717255737939626
    ```


### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[basicsr-dipnet-vdsp_params.json](../build_in/vdsp_params/basicsr-dipnet-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/basicsr_dipnet_int8/mod \
        --vdsp_params ../build_in/vdsp_params/basicsr-dipnet-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,1080,1920]
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path /path/to/DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --target_path  /path/to/DIV2K/DIV2K_valid_LR_bicubic/X2_npz \
        --text_path npz_datalist.txt
    ```
   
    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/basicsr_dipnet_int8/mod \
        --vdsp_params ../build_in/vdsp_params/basicsr-dipnet-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,1080,1920] \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度[basicsr-vamp_eval.py](../build_in/vdsp_params/basicsr-vamp_eval.py)，
    ```bash
    python ../build_in/vdsp_params/basicsr-vamp_eval.py \
        --gt_dir /path/to/DIV2K/DIV2K_valid_HR \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output \
        --input_shape 1080 1920 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```

