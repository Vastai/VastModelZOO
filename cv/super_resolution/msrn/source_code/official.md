## Official

### step.1 获取预训练模型

```
link: https://github.com/MIVRC/MSRN-PyTorch
branch: master
commit: a0e038de7eb42e21d2e88c38e6490b61a02c566e
```
- 拉取原始仓库：[RCAN_TestCode/code](https://github.com/MIVRC/MSRN-PyTorch)
- 将本仓库的[msrn.py](../source_code/official/model/msrn.py)和[common.py](../source_code/official/model/common.py)，移动到[MSRN/Train/model](https://github.com/MIVRC/MSRN-PyTorch/blob/master/MSRN/Train/model)目录，替换原始同名文件
- 参考[msrn.py](../source_code/official/model/msrn.py)，修改原始权重路径，导出torchscript和onnx

### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集

- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_msrn.yaml](../build_in/build/official_msrn.yaml)

    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd msrn
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_msrn.yaml
    ```

### step.4 模型推理

- 参考[official_vsx_inference.py](../build_in/vsx/python/official_vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/official_vsx_inference.py \
        --lr_image_dir  /path/to/DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --model_prefix_path deploy_weights/official_msrn_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-msrn-vdsp_params.json \
        --hr_image_dir /path/to/DIV2K/DIV2K_valid_HR \
        --save_dir ./infer_output \
        --device 0
    ```
    
    <details><summary>点击查看精度测试结果</summary>

    ```
    # fp16
    mean psnr: 31.810964798248452, mean ssim: 0.8790596313361325

    # int8 
    mean psnr: 31.94664468621006, mean ssim: 0.885171705512043
    ```

    </details>

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[official-msrn-vdsp_params.json](../build_in/vdsp_params/official-msrn-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/official_msrn_int8/mod \
    --vdsp_params ../build_in/vdsp_params/official-msrn-vdsp_params.json \
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
    vamp -m deploy_weights/official_msrn_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-msrn-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,1080,1920] \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[official-vamp_eval.py](../build_in/vdsp_params/official-vamp_eval.py)，
    ```bash
    python ../build_in/vdsp_params/official-vamp_eval.py \
        --gt_dir /path/to/DIV2K/DIV2K_valid_HR \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output/ \
        --input_shape 1080 1920 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```

