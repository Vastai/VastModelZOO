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
3. 性能测试，配置vdsp参数[official-msrn-vdsp_params.json](../vacc_code/vdsp_params/official-msrn-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/msrn-int8-max-3_1080_1920-vacc/mod \
    --vdsp_params ../vacc_code/vdsp_params/official-msrn-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,1080,1920]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/msrn-int8-max-3_1080_1920-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/official-msrn-vdsp_params.json \
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
    ./vamp -m deploy_weights/msrn_2x-int8-max-3_1080_1920-vacc/mod --vdsp_params official-msrn-vdsp_params.json -i 1 -p 1 -b 1
    model input shape 0: [3,1080,1920], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 92
    temperature (°C): 48.1079
    card power (W): 42.7576
    die memory used (MB): 1677.39
    throughput (qps): 24.3814
    e2e latency (us):
        avg latency: 102395
        min latency: 48197
        max latency: 129917
        p50 latency: 88914
        p90 latency: 123008
        p95 latency: 123049
        p99 latency: 123159
    model latency (us):
        avg latency: 102348
        min latency: 48154
        max latency: 129869
        p50 latency: 88846
        p90 latency: 122959
        p95 latency: 122998
        p99 latency: 123095

    # DIV2K Valid
    msrn_2x-3_1080_1920.torchscript.pt
    mean psnr: 32.599501072688085, mean ssim: 0.7719364672121912

    msrn_2x-int8-max-1_3_1080_1920-vacc
    mean psnr: 32.36808958001324, mean ssim: 0.7677706352576107

    ```
    </details>

### Tips
- 在`AI 1.5.2 SP1_0822`上验证，只能通过`int8 runstream`

