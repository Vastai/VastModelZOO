## Official

### step.1 获取预训练模型

```
link: https://github.com/yulunzhang/RCAN
branch: master
commit: 3339ebc59519c3bb2b5719b87dd36515ec7f3ba7
```
- 拉取原始仓库：[RCAN_TestCode/code](https://github.com/yulunzhang/RCAN/tree/master/RCAN_TestCode/code)
- 内部在原始模型基础上进行了修改，去除了通道注意力等模块，精简计算量，以适配VASTAI板卡
- 原始[rcan.py](https://github.com/yulunzhang/RCAN/tree/master/RCAN_TestCode/code/model/rcan.py)，修改后[rcan_modify.py](../source_code/official/model/rcan_modify.py)
- 原始[common.py](https://github.com/yulunzhang/RCAN/tree/master/RCAN_TestCode/code/model/common.py)，修改后[common_modify.py](../source_code/official/model/common_modify.py)
- 将上述两个修改后的文件放置在原始文件同目录下
- 将[export.py](../source_code/official/export.py)移动至`RCAN_TestCode/code`目录，修改原始权重路径，导出torchscript和onnx

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
3. 性能测试，配置vdsp参数[official-rcan-vdsp_params.json](../vacc_code/vdsp_params/official-rcan-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/rcan-int8-max-3_1080_1920-vacc/rcan \
    --vdsp_params ../vacc_code/vdsp_params/official-rcan-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/rcan-int8-max-3_1080_1920-vacc/rcan \
    --vdsp_params vacc_code/vdsp_params/official-rcan-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [official-vamp_eval.py](../vacc_code/vdsp_params/official-vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/official-vamp_eval.py \
    --gt_dir DIV2K/DIV2K_valid_HR \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/rcan \
    --input_shape 1080 1920 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```

- <details><summary>性能&精度</summary>

    ```
    ./vamp -m deploy_weights/RCAN-int8-max-3_1080_1920-vacc/RCAN --vdsp_params official-rcan-vdsp_params.json -i 8 -p 8 -b 1
    - number of instances in each device: 8
        devices: [0]
        batch szie: 1
        ai utilize (%): 84.7898
        temperature (°C): 48.9539
        card power (W): 31.4568
        die memory used (MB): 2962.27
        throughput (qps): 20.9133
        e2e latency (us):
        avg latency: 938804
        min latency: 242944
        max latency: 1778707
        p50 latency: 925637
        p90 latency: 1205204
        p95 latency: 1262033
        p99 latency: 1375009
        model latency (us):
        avg latency: 790979
        min latency: 241057
        max latency: 1396289
        p50 latency: 785185
        p90 latency: 933979
        p95 latency: 971597
        p99 latency: 1072986

    ./vamp_2.1.0 -m deploy_weights/RCAN2-int8-max-1_3_1080_1920-vacc/mod --vdsp_params official-rcan-vdsp_params.json -i 1 p 1 -b 1 
    model input shape 0: [3,1080,1920], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 94.381
    temperature (°C): 48.5141
    card power (W): 36.9088
    die memory used (MB): 1708.98
    throughput (qps): 36.1415
    e2e latency (us):
        avg latency: 69068
        min latency: 35048
        max latency: 90097
        p50 latency: 62426
        p90 latency: 83008
        p95 latency: 83048
        p99 latency: 83121
    model latency (us):
        avg latency: 69019
        min latency: 34996
        max latency: 90050
        p50 latency: 62369
        p90 latency: 82954
        p95 latency: 82991
        p99 latency: 83058

    # DIV2K
    RCAN-int8-max-3_1080_1920-vacc, 3 2 32
    mean psnr: 32.675621392364775, mean ssim: 0.7663607410574184

    RCAN2-int8-max-3_1080_1920-vacc, 2 2 16
    mean psnr: 32.349224814431395, mean ssim: 0.7642811655565711
    ```
    </details>

## Tips
- 原始仓库的模型均比较大，build会报错：已中断（系统内存问题）
- 修改版的RCAN内部定义为：SR4K
- 暂只提供2x放大
- 已验证大尺寸：input: [3, 1080, 1920]，output: [3, 2160, 3840] 2x with 4K
- 已验证大尺寸：input: [3, 2160, 3840]，output: [3, 4320, 7680] 2x with 8K，注意此输入下需占用内存24G，内存不够会在quant时报已中断
- fp16在runmodel和runstream下均无法跑通，会导致系统死机，需重启（PixelShuffle）
- runmodel int8 在[3, 1080, 1920]尺寸下build会报错，runstream ok


