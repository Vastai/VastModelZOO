## Official

```
link: https://github.com/xinntao/ESRGAN
branch: master
commit: 73e9b634cf987f5996ac2dd33f4050922398a921
```

### step.1 获取预训练模型
克隆原始仓库，将[export.py](./export.py)脚本放置在`{ESRGAN}`目录，执行以下命令，进行模型导出为onnx和torchscript：
```bash
# cd {ESRGAN}
python export.py
```

### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集，通过[image2npz.py](../../common/utils/image2npz.py)，转换为对应npz文件


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_esrgan.yaml](../build_in/build/official_esrgan.yaml)

    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd esrgan
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_esrgan.yaml
    ```

### step.4 模型推理

- 参考[vsx_inference.py](../build_in/vsx/python/vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/vsx_inference.py \
        --lr_image_dir  /path/to/DIV2K/DIV2K_valid_LR_bicubic/X4 \
        --model_prefix_path deploy_weights/official_esrgan_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-esrgan_x4-vdsp_params.json \
        --hr_image_dir /path/to/DIV2K/DIV2K_valid_HR \
        --save_dir ./infer_output \
        --device 0
    ```

    ```
    # fp16
    mean psnr: 14.732155245763165, mean ssim: 0.33772488257573435

    # int8 
    mean psnr: 14.340168813696778, mean ssim: 0.2378801214511364
    ```


### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[official-esrgan_x4-vdsp_params.json](../build_in/vdsp_params/official-esrgan_x4-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/official_esrgan_int8/mod \
    --vdsp_params ../build_in/vdsp_params/official-esrgan_x4-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128]
    ```


2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 准备数据，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path /path/to/DIV2K/DIV2K_valid_LR_bicubic/X4 \
        --target_path /path/to/DIV2K/DIV2K_valid_LR_bicubic/X4_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/official_esrgan_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-esrgan_x4-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,128,128] \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)，
    ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
        --gt_dir /path/to/DIV2K/DIV2K_valid_HR \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output \
        --input_shape 128 128 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```
