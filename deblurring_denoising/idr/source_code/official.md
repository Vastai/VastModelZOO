
### step.1 准备模型
```
link: https://github.com/zhangyi-3/IDR
branch: main
commit: f3f05d4bceff6ab780a841c6997c5daead859bda
```
- 克隆原始仓库
- 将文件[export.py](./export.py)移动至仓库目录，配置原始模型权重路径，执行导出onnx或torchscript


### step.2 准备数据集
- 下载[SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)数据集
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
    --gt_dir denoising/SIDD/val/target \
    --input_npz_path datalist_npz.txt \
    --out_npz_dir deploy_weights/idr_gaussian-int8-percentile-1_3_256_256-debug-result \
    --input_shape 256 256 \
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
    --dataset_path denoising/SIDD/val/lr_sigma_25 \
    --target_dir denoising/SIDD/val/lr_sigma_25_npz \
    --text_path datalist_npz.txt
    ```
3. 性能测试，配置vdsp参数[official-idr_gaussian-vdsp_params.json](../vacc_code/vdsp_params/official-idr_gaussian-vdsp_params.json)，执行：
    ```bash
    ./vamp -m deploy_weights/idr_gaussian-fp16-none-1_3_256_256_-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/official-idr_gaussian-vdsp_params.json \
    -i 1 -p 1 -b 1  -s [1,256,256]
    ```
4. 精度测试
    ```bash
    ./vamp -m deploy_weights/idr_gaussian-fp16-none-1_3_256_256_-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/official-idr_gaussian-vdsp_params.json \
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
    vamp_2.1.0 -mdeploy_weights/gaussian-fp16-none-1_3_256_256-vacc/mod --vdsp_params vdsp_params/official-idr_unet-vdsp_params.json -i 1 p 1 -b 1 
    model input shape 0: [3,256,256], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 95.0556
    temperature (°C): 48.7072
    card power (W): 63.3866
    die memory used (MB): 1196.73
    throughput (qps): 492.556
    e2e latency (us):
        avg latency: 4966
        min latency: 2580
        max latency: 6433
        p50 latency: 4484
        p90 latency: 5982
        p95 latency: 6005
        p99 latency: 6041
    model latency (us):
        avg latency: 4924
        min latency: 2528
        max latency: 6361
        p50 latency: 4411
        p90 latency: 5940
        p95 latency: 5960
        p99 latency: 5988


    ./vamp_2.1.0 -m deploy_weights/gaussian-int8-percentile-1_3_256_256-vacc/mod --vdsp_params vdsp_params/official-idr_unet-vdsp_params.json -i 1 p 1 -b 1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 94
    temperature (°C): 44.125
    card power (W): 60.2988
    die memory used (MB): 1164.85
    throughput (qps): 2078.2
    e2e latency (us):
        avg latency: 1134
        min latency: 838
        max latency: 1835
        p50 latency: 1251
        p90 latency: 1360
        p95 latency: 1378
        p99 latency: 1423
    model latency (us):
        avg latency: 1118
        min latency: 831
        max latency: 1823
        p50 latency: 1239
        p90 latency: 1338
        p95 latency: 1361
        p99 latency: 1399
    
    # SIDD/val

    gaussian.onnx
    mean psnr: 37.01257983158227, mean ssim: 0.8959340791629893

    deploy_weights/gaussian-fp16-none-1_3_256_256-vacc
    mean psnr: 37.01495209756972, mean ssim: 0.8959743214541712

    gaussian-int8-percentile-1_3_256_256-vacc
    mean psnr: 36.388936631858996, mean ssim: 0.8882093616956634
    ```
    </details>

