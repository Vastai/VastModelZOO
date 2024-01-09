
### step.1 准备模型
```
link: https://github.com/chosj95/mimo-unet
branch: main
commit: 5c580135ed1c03344ac9c741267324ff90b5f209
```
- 克隆原始仓库
- 原始模型forward返回列表，实际只需列表内最后元素，修改返回值
- 原始模型forward内上采样和下采样操作[F.interpolate](https://github.com/chosj95/MIMO-UNet/blob/main/models/MIMOUNet.py#L140)使用默认的`nearest`方式，在vdps中存在尺寸限制(3x360x640 int8 run 会报错)，替换为`bilinear`导出模型，可避免尺寸限制
- 上述修改后文件[MIMOUNet.py](./MIMOUNet.py)，替换原始仓库内的[models/MIMOUNet.py](https://github.com/chosj95/MIMO-UNet/blob/main/models/MIMOUNet.py)
- 将文件[export.py](./export.py)移动至仓库目录，配置原始模型权重路径，执行导出onnx或torchscript


### step.2 准备数据集
- 下载[GoPro](https://seungjunnah.github.io/Datasets/gopro)数据集，将测试集子文件夹图片合并至一起
- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件


### step.3 模型转换
1. 获取[vamc](../../../docs/doc_vamc.md)模型转换工具

2. 根据具体模型修改模型转换配置文件[config.yaml](../vacc_code/build/config.yaml)，执行转换命令：
    ```bash
    vamc build ../vacc_code/build/config.yaml
    ```

### step.4 模型推理
1. runmodel
   > `engine.type: debug`

  - 方式1：[sample_runmodel.py](../vacc_code/runmodel/sample_runmodel.py)，进行runmodel推理和eval评估
  - 方式2：也可使用vamc的run功能
    - 确保[config.yaml](../vacc_code/build/config.yaml)内dataset.path为验证集数据集路径，数据量为全部，执行以下命令后会在三件套同级目录下生成推理结果nzp目录
    ```bash
    vamc run ../vacc_code/build/config.yaml
    ```
    - 使用[vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)解析npz，绘图并统计精度（保证上面跑完全量的验证集）：
    ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --gt_dir GoPro/test/target \
    --input_npz_path datalist_npz.txt \
    --out_npz_dir deploy_weights/MIMO-UNet-fp16-none-1_3_360_640-debug-result \
    --input_shape 360 640 \
    --draw_dir npz_draw_result
    ```

2. vsx
   - [doc_vsx.md](../../../docs/doc_vsx.md)，参考文档安装推理`vsx`工具
   - [vsx_inference.py](../vacc_code/vsx/vsx_inference.py)，配置参数后执行，进行runstream推理和eval评估


### step.5 性能精度
1. 获取[vamp](../../../docs/doc_vamp.md)工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，生成`npz`格式的推理数据以及对应的`datalist_npz.txt`
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path GoPro/test/input \
    --target_dir GoPro/test/input_npz \
    --text_path datalist_npz.txt
    ```
3. 性能测试，配置vdsp参数[official-mimo_unet-vdsp_params.json](../vacc_code/vdsp_params/official-mimo_unet-vdsp_params.json)，执行：
    ```bash
    ./vamp -m deploy_weights/MIMO-UNet-fp16-none-1_3_360_640-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/official-mimo_unet-vdsp_params.json \
    -i 1 -p 1 -b 1  -s [1,360,640]
    ```
4. 精度测试
    ```bash
    ./vamp -m deploy_weights/MIMO-UNet-fp16-none-1_3_360_640-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/official-mimo_unet-vdsp_params.json \
    --datalist datalist_npz.txt \
    --path_output ./outputs \
    -i 1 -p 1 -b 1  -s [1,360,640]
    ```
5. 基于[vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析输出结果和精度评估
    ```bash
    python  ../vacc_code/vdsp_params/vamp_eval.py \
    --gt_dir GoPro/test/target \
    --input_npz_path datalist_npz.txt \
    --out_npz_dir outputs \
    --input_shape 360 640 \
    --draw_dir output_draw_result \
    --vamp_flag
    ```

- <details><summary>性能&精度</summary>

    ```
    ./vamp_2.1.0 -m MIMO-UNet-fp16-none-1_3_360_640-vacc/mod --vdsp_params official-mimo_unet-vdsp_params.json -i 1 p 1 -b 1 
    model input shape 0: [3,360,640], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 90.4339
    temperature (°C): 57.0485
    card power (W): 55.869
    die memory used (MB): 1420.88
    throughput (qps): 22.7082
    e2e latency (us):
        avg latency: 109891
        min latency: 45215
        max latency: 132954
        p50 latency: 89053
        p90 latency: 132213
        p95 latency: 132392
        p99 latency: 132679
    model latency (us):
        avg latency: 109838
        min latency: 45173
        max latency: 132903
        p50 latency: 89011
        p90 latency: 132157
        p95 latency: 132336
        p99 latency: 132631

    ./vamp_2.1.0 -m MIMO-UNet-int8-percentile-1_3_360_640-vacc/mod --vdsp_params official-mimo_unet-vdsp_params.json -i 1 p 1 -b 1 
    model input shape 0: [3,360,640], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 93.5311
    temperature (°C): 46.7177
    card power (W): 40.2804
    die memory used (MB): 1292.98
    throughput (qps): 29.7627
    e2e latency (us):
        avg latency: 83825
        min latency: 34574
        max latency: 101531
        p50 latency: 68048
        p90 latency: 100714
        p95 latency: 100764
        p99 latency: 100841
    model latency (us):
        avg latency: 83773
        min latency: 34522
        max latency: 101475
        p50 latency: 67995
        p90 latency: 100664
        p95 latency: 100703
        p99 latency: 100791

    ./vamp_2.1.0 -m MIMO-UNetPlus-fp16-none-1_3_360_640-vacc/mod --vdsp_params official-mimo_unet-vdsp_params.json -i 1 p 1 -b 1
    model input shape 0: [3,360,640], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 84.6422
    temperature (°C): 57.1006
    card power (W): 57.8757
    die memory used (MB): 1453.07
    throughput (qps): 11.5427
    e2e latency (us):
        avg latency: 216292
        min latency: 87562
        max latency: 261343
        p50 latency: 174403
        p90 latency: 260146
        p95 latency: 260406
        p99 latency: 260790
    model latency (us):
        avg latency: 216236
        min latency: 87511
        max latency: 261322
        p50 latency: 174292
        p90 latency: 260085
        p95 latency: 260353
        p99 latency: 260736

    ./vamp_2.1.0 -m MIMO-UNetPlus-int8-percentile-1_3_360_640-vacc/mod --vdsp_params official-mimo_unet-vdsp_params.json -i 1 p 1 -b 1outputs
    model input shape 0: [3,360,640], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 90.3441
    temperature (°C): 51.9829
    card power (W): 47.5735
    die memory used (MB): 1293.28
    throughput (qps): 22.7453
    e2e latency (us):
        avg latency: 109702
        min latency: 44942
        max latency: 132505
        p50 latency: 90311
        p90 latency: 131801
        p95 latency: 131850
        p99 latency: 131959
    model latency (us):
        avg latency: 109650
        min latency: 44933
        max latency: 132496
        p50 latency: 90245
        p90 latency: 131749
        p95 latency: 131797
        p99 latency: 131888
        
    GoPro/test
    weights/MIMO-UNet360_640dynamicbilinear.onnx
    mean psnr: 30.302147634134393, mean ssim: 0.9143153119050601

    MIMO-UNet-fp16-none-1_3_360_640-debug
    mean psnr: 29.857426955943, mean ssim: 0.9061482627584009

    MIMO-UNet-int8-percentile-1_3_360_640-debug
    mean psnr: 29.274464538812953, mean ssim: 0.8854495357031456

    MIMO-UNetPlus.onnx
    mean psnr: 30.85280045642696, mean ssim: 0.9240697168272072

    MIMO-UNetPlus-fp16-none-1_3_360_640-vacc
    mean psnr: 30.606055338442147, mean ssim: 0.9203058819660552

    MIMO-UNetPlus-int8-percentile-1_3_360_640-vacc
    mean psnr: 29.378983035610453, mean ssim: 0.8866139762188419
    ```
    </details>


## Tips
- 源模型中，上采样和下采样操作[F.interpolate](https://github.com/chosj95/MIMO-UNet/blob/main/models/MIMOUNet.py#L140)使用默认的`nearest`方式，在vdps中存在尺寸限制(3x360x640 int8 run 会报错)，替换为`bilinear`导出模型，可避免尺寸限制
