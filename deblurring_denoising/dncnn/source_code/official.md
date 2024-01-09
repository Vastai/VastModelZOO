
### step.1 准备模型
```
link: https://github.com/cszn/DnCNN
branch: master
commit: e93b27812d3ff523a3a79d19e5e50d233d7a8d0a
```

- 参考[export.py](./export.py)，导出onnx或torchscript


### step.2 准备数据集
- 下载[Set12](https://github.com/cszn/DnCNN/tree/master/testsets/Set12)数据集，通过[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，生成噪声图像和转换为对应npz文件

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
    --gt_dir Set12 \
    --input_npz_path datalist_npz.txt \
    --out_npz_dir deploy_weights/dncnn-int8-percentile-1_256_256-debug-result \
    --input_shape 256 256 \
    --draw_dir npz_draw_result
    ```

2. vsx
   - [doc_vsx.md](../../../docs/doc_vsx.md)，参考文档安装推理`vsx`工具
   - [vsx_inference.py](../vacc_code/vsx/vsx_inference.py)，配置参数后执行，进行runstream推理和eval评估


### step.5 性能精度
1. 获取[vamp](../../../docs/doc_vamp.md)工具
2. 基于[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，生成`npz`格式的推理数据以及对应的`datalist_npz.txt`
    ```bash
    python ../vacc_code/vdsp_params/image2npz.py \
    --source_dir testsets/Set12 \
    --noise_dir testsets/Set12_noise \
    --target_dir testsets/Set12_npz \
    --text_path datalist_npz.txt
    ```
3. 性能测试，配置vdsp参数[official-dncnn-vdsp_params.json](../vacc_code/vdsp_params/official-dncnn-vdsp_params.json)，执行：
    ```bash
    ./vamp -m deploy_weights/dncnn-fp16-none-1_256_256-vacc/dncnn \
    --vdsp_params vacc_code/vdsp_params/official-dncnn-vdsp_params.json \
    -i 1 -p 1 -b 1  -s [1,256,256]
    ```
4. 精度测试
    ```bash
    ./vamp -m deploy_weights/dncnn-fp16-none-1_256_256-vacc/dncnn \
    --vdsp_params vacc_code/vdsp_params/official-dncnn-vdsp_params.json \
    --datalist datalist_npz.txt \
    --path_output ./outputs/DnCNN \
    -i 1 -p 1 -b 1  -s [1,256,256]
    ```
5. 基于[vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析输出结果和精度评估
    ```bash
    python  ../vacc_code/vdsp_params/vamp_eval.py \
    --gt_dir testsets/Set12 \
    --input_npz_path datalist_npz.txt \
    --out_npz_dir outputs/DnCNN \
    --input_shape 256 256 \
    --draw_dir output_draw_result \
    --vamp_flag
    ```

- <details><summary>性能&精度</summary>

    ```
    
    ./vamp_2.1.0 -m deploy_weights/dncnn-int8-percentile-1_1_256_256-vacc/mod --vdsp_params vacc_code/vdsp_params/official-dncnn-vdsp_params.json -i 1 p 1 -b 1 
    model input shape 0: [3,256,256], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 94.4167
    temperature (°C): 48.5142
    card power (W): 54.7673
    die memory used (MB): 1164.75
    throughput (qps): 685.706
    e2e latency (us):
        avg latency: 3551
        min latency: 1444
        max latency: 5715
        p50 latency: 3559
        p90 latency: 5309
        p95 latency: 5347
        p99 latency: 5397
    model latency (us):
        avg latency: 3517
        min latency: 1389
        max latency: 5660
        p50 latency: 3514
        p90 latency: 5272
        p95 latency: 5302
        p99 latency: 5349

    ./vamp_2.1.0 -m deploy_weights/dncnn-fp16-none-1_1_256_256-vacc/mod --vdsp_params vdsp_params/official-dncnn-vdsp_params.json -i 1 p 1 -b 1 
    model input shape 0: [3,256,256], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 95.1017
    temperature (°C): 53.0476
    card power (W): 53.8038
    die memory used (MB): 1164.62
    throughput (qps): 152.71
    e2e latency (us):
        avg latency: 16251
        min latency: 4354
        max latency: 23884
        p50 latency: 15214
        p90 latency: 22786
        p95 latency: 22813
        p99 latency: 22859
    model latency (us):
        avg latency: 16210
        min latency: 4303
        max latency: 23799
        p50 latency: 15171
        p90 latency: 22736
        p95 latency: 22766
        p99 latency: 22805

    # Set12 256size sigma = 25
    
    # dncnn.onnx
    mean psnr : 29.68849189845346, mean ssim : 0.8465395489855134

    dncnn-fp16-none-1_1_256_256-debug
    mean psnr: 29.687818679216495, mean ssim: 0.8464874660788763

    dncnn-int8-percentile-1_1_256_256-debug
    mean psnr: 29.57925011136118, mean ssim: 0.8373199017065495
    ```
    </details>


## Tips
- 模型输入为灰度图，1x256x256
- 模型的输入是含噪声图像：对无噪声图像增加高斯白噪声，构造含噪声图像
