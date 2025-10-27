## ECBSR

### step.1 获取预训练模型

```
link: https://github.com/XPixelGroup/BasicSR
branch: v1.4.2
commit: 651835a1b9d38dbbdaf45750f56906be2364f01a
```
- 拉取原始仓库，使用配置表[train_ECBSR_x4_m4c16_prelu_RGB_mini.yml](../source_code/basicsr/train_ECBSR_x4_m4c16_prelu_RGB_mini.yml)，训练模型
- 将[export.py](../source_code/basicsr/export.py)放置于{BasicSR}仓库根目录，配置权重路径，执行即可到torchscript和onnx


### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集

- 通过[image2npz.py](../build_in/vsx/python/image2npz.py)，将测试低清LR图像转换为对应npz文件

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [basicsr_ecbsr.yaml](../build_in/build/basicsr_ecbsr.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd ecbsr
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/basicsr_ecbsr.yaml
    ```

### step.4 模型推理

1. runstream
    - 参考[vsx脚本](../build_in/vsx/python/basicsr_vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/basicsr_vsx_inference.py \
        --lr_image_dir  /path/to/DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --model_prefix_path deploy_weights/basicsr_ecbsr_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/basicsr-ecbsr_rgb_2x-vdsp_params.json \
        --hr_image_dir /path/to/DIV2K/DIV2K_valid_HR \
        --save_dir ./runstream_output \
        --device 0
    ```

    ```
    # 模型名：ecbsr_rgb_2x-3_1080_1920

    # fp16
    mean psnr: 31.1576080968868, mean ssim: 0.8600484183986762

    # int8 
    mean psnr: 31.680397037643402, mean ssim: 0.8495486294516128
    ```


### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[basicsr-ecbsr_rgb_2x-vdsp_params.json](../build_in/vdsp_params/basicsr-ecbsr_rgb_2x-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/basicsr_ecbsr_run_stream_int8/mod \
        --vdsp_params ../build_in/vdsp_params/basicsr-ecbsr_rgb_2x-vdsp_params.json \
        -i 1 p 1 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致
    
    - 数据准备，基于[image2npz.py](../build_in/vsx/python/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../build_in/vsx/python/image2npz.py \
        --dataset_path /path/to/DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --target_path  /path/to/DIV2K/DIV2K_valid_LR_bicubic/X2_npz \
        --text_path npz_datalist.txt
    ```
    
    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/basicsr_ecbsr_run_stream_int8/mod \
        --vdsp_params ../build_in/vdsp_params/basicsr-ecbsr_rgb_2x-vdsp_params.json \
        -i 1 p 1 -b 1 \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[basicsr-vamp_eval.py](../build_in/vdsp_params/basicsr-vamp_eval.py)，
    ```bash
    python ../build_in/vdsp_params/basicsr-vamp_eval.py \
        --gt_dir /path/to/DIV2K/DIV2K_valid_HR \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output \
        --input_shape 1080 1920 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```
