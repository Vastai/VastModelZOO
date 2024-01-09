## Official

### step.1 获取预训练模型

```
link: https://github.com/Algolzw/NCNet
branch: main
commit: 048486534a209c72ef6bbe991a4a926e61c18345
```
- [export.py](../source_code/official/export.py)，修改原始权重路径，导出torchscript和onnx

### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集

- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件


### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改模型转换配置文件[](../vacc_code/build/official-config.yaml)，执行转换命令：
    ```bash
    vamc build ../vacc_code/build/official-config.yaml
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
3. 性能测试，配置vdsp参数[official-ncnet-vdsp_params.json](../vacc_code/vdsp_params/official-ncnet-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/ncnet-int8-max-3_1080_1920-vacc/mod \
    --vdsp_params ../vacc_code/vdsp_params/official-ncnet-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/ncnet-int8-max-3_1080_1920-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/official-ncnet-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [official-vamp_eval.py](../vacc_code/vdsp_params/official-vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/official-vamp_eval.py \
    --gt_dir DIV2K/DIV2K_valid_HR \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/ncnet \
    --input_shape 1080 1920 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```

- <details><summary>性能&精度</summary>

    ```
    ./vamp_2.1.0 -m /home/simplew/code/vastmodelzoo/0818/algorithm_modelzoo/deploy_weights/ncnet_model_best-int8-max-1_3_1080_1920-vacc/mod --
    vdsp_params /home/simplew/code/vastmodelzoo/0818/algorithm_modelzoo/super_resolution/ncnet/vacc_code/vdsp_params/official-ncnet-vdsp_params.json -i 1 p 1 -b 1 
    model input shape 0: [3,1080,1920], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 93.5597
    temperature (°C): 46.2237
    card power (W): 36.3562
    die memory used (MB): 1676.69
    throughput (qps): 34.1024
    e2e latency (us):
        avg latency: 73148
        min latency: 36641
        max latency: 95027
        p50 latency: 65792
        p90 latency: 87857
        p95 latency: 87884
        p99 latency: 87957
    model latency (us):
        avg latency: 73101
        min latency: 36589
        max latency: 94976
        p50 latency: 65737
        p90 latency: 87807
        p95 latency: 87835
        p99 latency: 87893

    # DIV2K
    ncnet_model_best.torchscript.pt
    mean psnr: 32.91744656201368, mean ssim: 0.7757775731518973

    ncnet_model_best-int8-max-1_3_1080_1920-vacc
    mean psnr: 32.448221007387524, mean ssim: 0.7663422973519197
    ```
    </details>

## Tips
- 验证版本`AI 1.5.2 SP1_0822`
- torch版本
  - int8，runstream build&run ok
  - int8，runmodel build报段错误
  - fp16，runmodel&runstream build报错


