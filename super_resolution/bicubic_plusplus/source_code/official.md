## Official

### step.1 获取预训练模型

```
link: https://github.com/aselsan-research-imaging-team/bicubic-plusplus
branch: main
commit: 52084c8d016f5e8a5cd62c050e47536a0d022177
```
- 参考[export.py](../source_code/official/export.py)，修改原始权重路径，导出torchscript和onnx


### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集

- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件



### step.3 模型转换
1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[official-config.yaml](../vacc_code/build/official-config.yaml)，执行转换命令：
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
3. 性能测试，配置vdsp参数[official-bicubic_plusplus_2x-vdsp_params.json](../vacc_code/vdsp_params/official-bicubic_plusplus_2x-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/bicubic_plusplus_2x-int8-kl_divergence-3_1080_1920-vacc/mod \
    --vdsp_params ../vacc_code/vdsp_params/official-bicubic_plusplus_2x-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/bicubic_plusplus_2x-int8-kl_divergence-3_1080_1920-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/official-bicubic_plusplus_2x-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [official-vamp_eval.py](../vacc_code/vdsp_params/official-vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/official-vamp_eval.py \
    --gt_dir DIV2K/DIV2K_valid_HR \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/ \
    --input_shape 1080 1920 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```

- <details><summary>性能&精度</summary>

    ```
    ./vamp_2.1.0 -m bicubic_plusplus_3x-int8-percentile-1_3_1080_1920-vacc/mod --vdsp_params official-bicubic_plusplus_3x-vdsp_params.json -i 1 p 1 -b 1
    model input shape 0: [3,1080,1920], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 88.6355
    temperature (°C): 46.2209
    card power (W): 32.5533
    die memory used (MB): 2332.59
    throughput (qps): 10.4588
    e2e latency (us):
        avg latency: 238786
        min latency: 109245
        max latency: 300206
        p50 latency: 204450
        p90 latency: 286836
        p95 latency: 286889
        p99 latency: 286991
    model latency (us):
        avg latency: 238732
        min latency: 109193
        max latency: 300152
        p50 latency: 204388
        p90 latency: 286787
        p95 latency: 286831
        p99 latency: 286928

    ./vamp_2.1.0 -m bicubic_plusplus_2x-int8-percentile-1_3_1080_1920-vacc/mod --vdsp_params official-bicubic_plusplus_2x-vdsp_params.json -i 1 p 1 -b 1
    model input shape 0: [3,1080,1920], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 94.8172
    temperature (°C): 47.3399
    card power (W): 37.0211
    die memory used (MB): 1644.59
    throughput (qps): 97.2483
    e2e latency (us):
        avg latency: 25578
        min latency: 18515
        max latency: 38818
        p50 latency: 28624
        p90 latency: 30722
        p95 latency: 30754
        p99 latency: 30814
    model latency (us):
        avg latency: 25536
        min latency: 18460
        max latency: 38768
        p50 latency: 28553
        p90 latency: 30678
        p95 latency: 30706
        p99 latency: 30760

    # DIV2K_valid_LR_bicubic/X3/
    bicubic_plusplus_3x.torchscript.pt
    mean psnr: 29.358160814986014, mean ssim: 0.8852686975501175
    bicubic_plusplus_3x-int8-percentile-1_3_1080_1920-vacc
    mean psnr: 28.2104789043819, mean ssim: 0.7914121409854921
    
    # DIV2K_valid_LR_bicubic/X2/
    bicubic_plusplus_2x.torchscript.pt
    mean psnr: 32.91128643538906, mean ssim: 0.7753685239469096
    bicubic_plusplus_2x-int8-kl_divergence-1_3_1080_1920-vacc
    mean psnr: 31.559440054744663, mean ssim: 0.7512467713689873
    ```
    </details>

