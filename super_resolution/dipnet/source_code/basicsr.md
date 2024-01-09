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
1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[basicsr-config.yaml](../vacc_code/build/basicsr-config.yaml)，执行转换命令：
    ```bash
    vamc build ../vacc_code/build/basicsr-config.yaml
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
3. 性能测试，配置vdsp参数[basicsr-dipnet-vdsp_params.json](../vacc_code/vdsp_params/basicsr-dipnet-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/dipnet-int8-max-3_1080_1920-vacc/mod \
    --vdsp_params ../vacc_code/vdsp_params/basicsr-dipnet-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/dipnet-int8-max-3_1080_1920-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/basicsr-dipnet-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [basicsr-vamp_eval.py](../vacc_code/vdsp_params/basicsr-vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/basicsr-vamp_eval.py \
    --gt_dir DIV2K/DIV2K_valid_HR \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/ \
    --input_shape 1080 1920 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```

- <details><summary>性能&精度</summary>

    ```
    ./vamp_2.1.0 -m deploy_weights/dipnet_x2_noSE_mini_net_g_368000-int8-percentile-1_3_1080_1920-vacc/mod --vdsp_params super_resolution/dipnet/vacc_code/vdsp_params/basicsr-dipnet-vdsp_params.json -i 1 p 1 -b 1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 91.6641
    temperature (°C): 49.5416
    card power (W): 45.3943
    die memory used (MB): 1678.05
    throughput (qps): 23.7118
    e2e latency (us):
        avg latency: 105270
        min latency: 49422
        max latency: 133523
        p50 latency: 91416
        p90 latency: 126463
        p95 latency: 126500
        p99 latency: 126589
    model latency (us):
        avg latency: 105222
        min latency: 49373
        max latency: 133477
        p50 latency: 91360
        p90 latency: 126410
        p95 latency: 126447
        p99 latency: 126531

    # DIV2K Valid
    dipnet_x2_noSE_mini_net_g_368000.torchscript.pt
    mean psnr: 32.91219246022892, mean ssim: 0.7761552386115719

    dipnet_x2_noSE_mini_net_g_368000-int8-percentile-1_3_1080_1920-vacc
    mean psnr: 32.44883277956704, mean ssim: 0.7696349512738535

    ```
    </details>

