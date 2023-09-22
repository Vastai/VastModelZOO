## Official

### step.1 获取预训练模型

```
link: https://github.com/xindongzhang/ECBSR
branch: main
commit: c60b1d2712af4ed4a615c2b7afa3980222c44a31
```
- 克隆原始仓库
- 将文件[export.py](../source_code/official/export.py)移动至原始仓库根目录；修改原始权重路径，导出torchscript和onnx


### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集

- 通过[image2npz.py](../vacc_code/vsx/image2npz.py)，将测试低清LR图像转换为对应npz文件(注意此过程，BGR-RGB-YCbCR-Y)



### step.3 模型转换
1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[official-config.yaml](../vacc_code/build/official-config.yaml)，执行转换命令：
    ```bash
    vamc build ../vacc_code/build/official-config.yaml
    ```

### step.4 benchmark

1. 获取vamp性能测试工具
2. 基于[image2npz.py](../vacc_code/vsx/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path DIV2K/DIV2K_valid_LR_bicubic/X2 \
    --target_path  DIV2K/DIV2K_valid_LR_bicubic/X2_Y_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[official-ecbsr_plain_2x-vdsp_params.json](../vacc_code/vdsp_params/official-ecbsr_plain_2x-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/ecbsr_plain_2x-int8-percentile-3_1080_1920-vacc/mod \
    --vdsp_params ../vacc_code/vdsp_params/official-ecbsr_plain_2x-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/ecbsr_plain_2x-int8-percentile-3_1080_1920-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/official-ecbsr_plain_2x-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [official-vamp_eval.py](../vacc_code/vdsp_params/official-vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/official-vamp_eval.py \
    --gt_dir DIV2K/DIV2K_valid_HR \
    --lr_dir DIV2K/DIV2K_valid_LR_bicubic/X2 \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/ \
    --input_shape 1080 1920 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```

- <details><summary>性能&精度</summary>

    ```
    ./vamp_2.1.0 -m deploy_weights/ecbsr_plain_2x-int8-percentile-1_1_1080_1920-vacc/mod --vdsp_params vdsp_params/official-plain-vdsp_params.json -i 1 -p 1 -b 1
    model input shape 0: [3,1080,1920], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 94.9018
    temperature (°C): 49.1461
    card power (W): 31.4001
    die memory used (MB): 1516.29
    throughput (qps): 81.5821
    e2e latency (us):
        avg latency: 30608
        min latency: 15915
        max latency: 40200
        p50 latency: 27915
        p90 latency: 36888
        p95 latency: 36942
        p99 latency: 37039
    model latency (us):
        avg latency: 30561
        min latency: 15873
        max latency: 40153
        p50 latency: 27849
        p90 latency: 36840
        p95 latency: 36894
        p99 latency: 36984

    ./vamp_2.1.0 -m deploy_weights/ecbsr_plain_2x-fp16-none-1_1_1080_1920-vacc/mod --vdsp_params vdsp_params/official-plain-vdsp_params.json -i 1 -p 1 -b 1
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 94.947
    temperature (°C): 49.7054
    card power (W): 34.19
    die memory used (MB): 1580.57
    throughput (qps): 69.1167
    e2e latency (us):
        avg latency: 36204
        min latency: 18866
        max latency: 47633
        p50 latency: 42401
        p90 latency: 43846
        p95 latency: 43953
        p99 latency: 44222
    model latency (us):
        avg latency: 36157
        min latency: 18807
        max latency: 47579
        p50 latency: 42347
        p90 latency: 43798
        p95 latency: 43925
        p99 latency: 44190


    # DIV2K_valid_LR_bicubic/X2/
    # 原始模型
    ecbsr_2x-1080_1920.torchscript.pt
    mean psnr: 32.59298674375998, mean ssim: 0.7749856646062577

    ecbsr_2x-fp16-none-1_1_1080_1920-vacc
    mean psnr: 31.583973032954074, mean ssim: 0.750445889684897
    ecbsr_2x-int8-percentile-1_1_1080_1920-vacc
    mean psnr: 31.50376265871793, mean ssim: 0.744641008963285

    # 重参数化后模型
    ecbsr_plain_2x-1080_1920.torchscript.pt
    mean psnr: 32.59538342872409, mean ssim: 0.7748545753681958

    ecbsr_plain_2x-fp16-none-1_1_1080_1920-vacc
    mean psnr: 32.30710768597619, mean ssim: 0.7740473069736203
    ecbsr_plain_2x-int8-percentile-1_1_1080_1920-vacc
    mean psnr: 32.34440159638396, mean ssim: 0.7689059544343012
    ```
    </details>

### Tips

- 与其它大尺寸超分模型不同，`enable_graph_partition`量化参数`不能`设置为True，会报错
