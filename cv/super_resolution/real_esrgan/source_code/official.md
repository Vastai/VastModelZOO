## Official

```
link: https://github.com/xinntao/Real-ESRGAN
branch: master
commit: 5ca1078535923d485892caee7d7804380bfc87fd
```

### step.1 获取预训练模型
克隆原始仓库，参考[pytorch2onnx.py](https://github.com/xinntao/Real-ESRGAN/blob/master/scripts/pytorch2onnx.py)，将[export.py](./export.py)脚本放置在前述脚本相同目录，执行以下命令，进行模型导出为onnx和torchscript：
```bash
# cd {Real-ESRGAN}
python scripts/export.py
```

### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集，通过[image2npz.py](../../common/utils/image2npz.py)，转换为对应npz文件

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_real_esrgan.yaml](../build_in/build/official_real_esrgan.yaml)
        
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 执行转换
    ```bash
    cd real_esrgan
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_real_esrgan.yaml
    ```

### step.4 模型推理

    - 参考: [official_vsx_inference.py](../build_in/vsx/python/official_vsx_inference.py)
    
    ```bash
    python ../build_in/vsx/python/official_vsx_inference.py \
        --lr_image_dir  /path/to/DIV2K/DIV2K_valid_LR_bicubic/X4 \
        --model_prefix_path deploy_weights/official_real_esrgan_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-real_esrgan_x4plus-vdsp_params.json \
        --hr_image_dir /path/to/DIV2K/DIV2K_valid_HR \
        --save_dir ./infer_output \
        --device 0
    ```
    - 精度结果在打印信息最后，如下：
    ```
    # fp16
    mean psnr: 20.595832061408863, mean ssim: 0.5732575982239713

    # int8
    mean psnr: 20.253986951966297, mean ssim: 0.5353068184362612
    ```

### step.5 性能精度测试
1. 性能测试，
    - 配置vdsp参数[official-real_esrgan_x4plus-vdsp_params.json](../build_in/vdsp_params/official-real_esrgan_x4plus-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/official_real_esrgan_int8/mod \
    --vdsp_params ../build_in/vdsp_params/official-real_esrgan_x4plus-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128]
    ```
    
2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path /path/to/DIV2K/DIV2K_valid_LR_bicubic/X4 \
        --target_path /path/to/DIV2K/DIV2K_valid_LR_bicubic/X4_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理，获得npz结果
    ```bash
    vamp -m deploy_weights/official_real_esrgan_int8/mod \
    --vdsp_params build_in/vdsp_params/official-real_esrgan_x4plus-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```

    - 解析npz结果，统计精度：[vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)
   ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
        --gt_dir /path/to/DIV2K/DIV2K_valid_HR \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output \
        --input_shape 128 128 \
        --draw_dir npz_draw_result \
        --vamp_flag
   ```
