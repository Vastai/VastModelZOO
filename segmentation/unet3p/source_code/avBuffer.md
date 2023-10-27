## avBuffer

```
link: https://github.com/avBuffer/UNet3plus_pth
branch: master
commit: 263534e4a48964e907324622b14b90f1c3b4270d
```

### step.1 获取预训练模型
- 克隆原始仓库
- 将文件[export.py](../source_code/export.py)移动至仓库根目录；修改其相关参数，即可导出torchscript和onnx



### step.2 准备数据集
- 原仓库中下载[Deep Automatic Portrait Matting](https://github.com/avBuffer/UNet3plus_pth)数据集


### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[config.yaml](../vacc_code/build/config.yaml)
3. 执行转换

   ```bash
   vamc build ../vacc_code/build/config.yaml
   ```

### step.4 benchmark
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式（注意配置图片后缀为`.png`）：
    ```bash
    python ../vacc_code/vdsp_params/image2npz.py \
    --dataset_path AutoPortraitMatting/testing/images \
    --target_path  AutoPortraitMatting/testing/images_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[avbuffer-unetpp-vdsp_params.json](../vacc_code/vdsp_params/avbuffer-unetpp-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/unet3p-int8-kl_divergence-1_3_128_128-vacc/mod \
    --vdsp_params ../vacc_code/vdsp_params/avbuffer-unetpp-vdsp_params.json \
    -i 1 p 1 -b 1
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/unet3p-int8-kl_divergence-1_3_128_128-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/avbuffer-unetpp-vdsp_params.json \
    -i 1 p 1 -b 1 \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --src_dir AutoPortraitMatting/testing/images \
    --gt_dir AutoPortraitMatting/testing/masks \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir vamp/opuputs \
    --input_shape 128 128 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


- <details><summary>性能&精度</summary>

    ```
    ./vamp_2.1.0 -m deploy_weights/unet3p-int8-kl_divergence-1_3_128_128-vacc/mod --vdsp_params vdsp_params/unetzoo-unetpp-vdsp_params.json -i 1 -b 1 -p 1
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 95.0294
    temperature (°C): 46.3129
    card power (W): 35.6416
    die memory used (MB): 1196.74
    throughput (qps): 261.411
    e2e latency (us):
        avg latency: 9429
        min latency: 4292
        max latency: 11732
        p50 latency: 7958
        p90 latency: 11353
        p95 latency: 11381
        p99 latency: 11442
    model latency (us):
        avg latency: 9385
        min latency: 4256
        max latency: 11681
        p50 latency: 7894
        p90 latency: 11306
        p95 latency: 11330
        p99 latency: 11383

    ./vamp_2.1.0 -m deploy_weights/unet3p_deepsupervision-int8-kl_divergence-1_3_128_128-vacc/mod --vdsp_params vdsp_params/unetzoo-unet3p_deepsupervision-vdsp_params.json -i 1 -b 1 -p 1
    model input shape 0: [3,128,128], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 94.9512
    temperature (°C): 46.3605
    card power (W): 34.3433
    die memory used (MB): 1196.74
    throughput (qps): 221.447
    e2e latency (us):
        avg latency: 11148
        min latency: 5144
        max latency: 13947
        p50 latency: 9436
        p90 latency: 13428
        p95 latency: 13492
        p99 latency: 13607
    model latency (us):
        avg latency: 11100
        min latency: 5080
        max latency: 13887
        p50 latency: 9372
        p90 latency: 13384
        p95 latency: 13442
        p99 latency: 13548

    # AutoPortraitMatting dataste
    unet3p-128.onnx
    mean iou: 0.8028378863260259

    unet3p-fp16-none-1_3_128_128-debug
    mean iou: 0.79324814183556
    unet3p-int8-kl_divergence-1_3_128_128-debug
    mean iou: 0.7935774424199835

    unet3p_deepsupervision-128.onnx
    mean iou: 0.7141104264658364

    unet3p_deepsupervision-fp16-none-1_3_128_128-debug
    mean iou: 0.7110279268652795
    unet3p_deepsupervision-int8-kl_divergence-1_3_128_128-debug
    mean iou: 0.70822689047088
    ```
    </details>


### Tips
- 原始仓库未提供预训练权重，基于AutoPortraitMatting自训练了两个子模型，精度一般
