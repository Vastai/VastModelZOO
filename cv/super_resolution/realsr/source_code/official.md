## Official

```
link: https://github.com/Tencent/Real-SR
branch: master
commit: 10276f4e497894dcbaa6b4244c06bba881b9460c
```

### step.1 获取预训练模型
克隆原始仓库，将[export.py](./export.py)脚本放置在源仓库目录，执行以下命令，进行模型导出为onnx和torchscript。
```bash
# cd {Real-SR/code}
python export.py
```

### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集([Validation Data (HR images)](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip))，通过[image2npz.py](../build_in/vdsp_params/image2npz.py)，转换为HR和LR及对应npz文件



### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_realsr.yaml](../build_in/build/official_realsr.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd realsr
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_realsr.yaml
    ```

### step.4 模型推理
1. runstream
    - 参考[official_vsx_inference.py](../build_in/vsx/python/official_vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/official_vsx_inference.py \
        --lr_image_dir  /path/to/realsr/DIV2K_valid_LR_128 \
        --model_prefix_path deploy_weights/official_realsr_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-realsr_DF2K-vdsp_params.json \
        --hr_image_dir /path/to/realsr/DIV2K_valid_HR_512 \
        --save_dir ./runstream_output \
        --device 0
    ```
    
    ```
    # fp16
    mean psnr: 23.65672187915699, mean ssim: 0.6818684566451145

    # int8 
    mean psnr: 23.182149272247926, mean ssim: 0.6331746012455065
    ```


### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[official-realsr_DF2K-vdsp_params.json](../build_in/vdsp_params/official-realsr_DF2K-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_realsr_run_stream_int8/mod \
    --vdsp_params ../build_in/vdsp_params/official-realsr_DF2K-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128]
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../build_in/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../build_in/vdsp_params/image2npz.py \
        --dataset_path /path/to/realsr/DIV2K_valid_HR \
        --hr_path /path/to/realsr/DIV2K_valid_HR_512 \
        --lr_path /path/to/realsr/DIV2K_valid_LR_128 \
        --target_path /path/to/realsr/DIV2K_valid_LR_128_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/official_realsr_run_stream_int8/mod \
        --vdsp_params build_in/vdsp_params/official-DF2K-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,128,128] \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)，
    ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
        --gt_dir /path/to/realsr/DIV2K_valid_HR_512 \
        --input_npz_path npz_datalist.txt \
        --out_npz_dirnpz_output \
        --input_shape 128 128 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```

