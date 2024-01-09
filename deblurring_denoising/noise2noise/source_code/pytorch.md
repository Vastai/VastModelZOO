
### step.1 准备模型
```
# pytorch
link: https://github.com/joeylitalien/noise2noise-pytorch
branch: master
commit: 1a284a1a1c9db123e43b32e3f8bce277c5ca7b3b
```
- 克隆原始仓库
- 将文件[export.py](./export.py)移动至仓库{noise2noise-pytorch/src}目录下，配置原始模型权重路径，执行导出onnx或torchscript


### step.2 准备数据集
- 下载[SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)数据集
- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件

### step.3 模型转换
1. 获取[vamc](../../../docs/doc_vamc.md)模型转换工具

2. 根据具体模型修改模型转换配置文件[pytorch_config.yaml](../vacc_code/build/pytorch_config.yaml)，执行转换命令：
    ```bash
    vamc build ../vacc_code/build/pytorch_config.yaml
    ```

### step.4 模型推理
1. runmodel
   > `engine.type: debug`

  - 方式1：[pytorch_sample_runmodel.py](../vacc_code/runmodel/pytorch_sample_runmodel.py)，进行runmodel推理和eval评估
  - 方式2：也可使用vamc的run功能
    - 确保[pytorch_config.yaml](../vacc_code/build/pytorch_config.yaml)内dataset.path为验证集数据集路径，数据量为全部，执行以下命令后会在三件套同级目录下生成推理结果nzp目录
    ```bash
    vamc run ../vacc_code/build/pytorch_config.yaml
    ```
    - 使用[pytorch_vamp_eval.py](../vacc_code/vdsp_params/pytorch_vamp_eval.py)解析npz，绘图并统计精度（保证上面跑完全量的验证集）：
    ```bash
    python ../vacc_code/vdsp_params/pytorch_vamp_eval.py \
    --gt_dir denoising/SIDD/val/target \
    --input_npz_path datalist_npz.txt \
    --out_npz_dir deploy_weights/noise2noise_gaussian-int8-percentile-1_3_256_256-debug-result \
    --input_shape 256 256 \
    --draw_dir npz_draw_result
    ```

2. vsx
   - [doc_vsx.md](../../../docs/doc_vsx.md)，参考文档安装推理`vsx`工具
   - [pytorch_vsx_inference.py](../vacc_code/vsx/pytorch_vsx_inference.py)，配置参数后执行，进行runstream推理和eval评估


### step.5 性能精度
1. 获取[vamp](../../../docs/doc_vamp.md)工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，生成`npz`格式的推理数据以及对应的`datalist_npz.txt`
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path denoising/SIDD/val/lr_sigma_25 \
    --target_dir denoising/SIDD/val/lr_sigma_25_npz \
    --text_path datalist_npz.txt
    ```
3. 性能测试，配置vdsp参数[pytorch-noise2noise_gaussian-vdsp_params.json](../vacc_code/vdsp_params/pytorch-noise2noise_gaussian-vdsp_params.json)，执行：
    ```bash
    ./vamp -m deploy_weights/noise2noise_gaussian-fp16-none-1_3_256_256_-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/pytorch-noise2noise_gaussian-vdsp_params.json \
    -i 1 -p 1 -b 1  -s [1,256,256]
    ```
4. 精度测试
    ```bash
    ./vamp -m deploy_weights/noise2noise_gaussian-fp16-none-1_3_256_256_-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/pytorch-noise2noise_gaussian-vdsp_params.json \
    --datalist datalist_npz.txt \
    --path_output ./outputs \
    -i 1 -p 1 -b 1  -s [1,256,256]
    ```
5. 基于[vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析输出结果和精度评估
    ```bash
    python  ../vacc_code/vdsp_params/vamp_eval.py \
    --gt_dir denoising/SIDD/val/target \
    --input_npz_path datalist_npz.txt \
    --out_npz_dir outputs \
    --input_shape 256 256 \
    --draw_dir output_draw_result \
    --vamp_flag
    ```

- <details><summary>性能&精度</summary>

    ```
    ./vamp_2.1.0 -m deploy_weights/gaussian_dynamic-fp16-none-1_3_256_256-vacc/mod --vdsp_params vdsp_params/pytorch-noise2noise-vdsp_params.json -i 1 p 1 -b 1
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 95.125
    temperature (°C): 48.9519
    card power (W): 64.3675
    die memory used (MB): 1164.73
    throughput (qps): 533.794
    e2e latency (us):
        avg latency: 4587
        min latency: 2458
        max latency: 6020
        p50 latency: 4229
        p90 latency: 5528
        p95 latency: 5556
        p99 latency: 5608
    model latency (us):
        avg latency: 4541
        min latency: 2404
        max latency: 5953
        p50 latency: 4166
        p90 latency: 5481
        p95 latency: 5503
        p99 latency: 5551


    ./vamp_2.1.0 -m deploy_weights/gaussian_dynamic-int8-kl_divergence-1_3_256_256-vacc/mod --vdsp_params vdsp_params/pytorch-noise2noise-vdsp_params.json -i 1 p 1 -b 1
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 97.25
    temperature (°C): 44.03
    card power (W): 60.2181
    die memory used (MB): 1164.99
    throughput (qps): 1840.61
    e2e latency (us):
        avg latency: 1291
        min latency: 952
        max latency: 4227
        p50 latency: 1395
        p90 latency: 1558
        p95 latency: 1580
        p99 latency: 1619
    model latency (us):
        avg latency: 1278
        min latency: 940
        max latency: 4210
        p50 latency: 1390
        p90 latency: 1543
        p95 latency: 1564
        p99 latency: 1594
    
    # SIDD/val gaussian sigma 25

    ckpts/gaussian/n2n-gaussian_dynamic.onnx
    mean psnr: 37.150769711965985, mean ssim: 0.8996455530899421

    gaussian_dynamic-fp16-none-1_3_256_256-debug
    mean psnr: 37.149148284444564, mean ssim: 0.8996432407421493

    gaussian_dynamic-int8-kl_divergence-1_3_256_256-debug
    mean psnr: 36.820258837760285, mean ssim: 0.8956484389711814
    ```
    </details>
