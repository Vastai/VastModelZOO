
### step.1 准备模型
```
link: https://github.com/megvii-model/HINet
branch: main
commit: 4e7231543090e6280d03fac22b3bb6869a25dfad
```

- 克隆原始仓库，将本地[export.py](./export.py)文件放置于[basicsr](https://github.com/megvii-model/HINet/tree/main/basicsr)文件夹内
- 修改模型forward返回值，只返回最后一个值，将[basicsr/models/archs/hinet_arch.py#L114](https://github.com/megvii-model/HINet/blob/main/basicsr/models/archs/hinet_arch.py#L114)，修改为`return out_2`
- 参考[export.py](./export.py)，导出onnx或torchscript


### step.2 准备数据集

- 下载[SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)数据集

- 下载[GoPro](https://seungjunnah.github.io/Datasets/gopro)数据集，将测试集子文件夹图片合并至一起

- 下载[Rain13K/Test100](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs)数据集

- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件

### step.3 模型转换
1. 获取[vamc](../../../docs/doc_vamc.md)模型转换工具

2. 根据具体模型修改模型转换配置文件[basicsr-config.yaml](../vacc_code/build/basicsr-config.yaml)，执行转换命令：
    ```bash
    vamc build ../vacc_code/build/basicsr-config.yaml
    ```

### step.4 模型推理
1. runmodel
   > `engine.type: debug`

  - 方式1：[basicsr-sample_runmodel.py](../vacc_code/runmodel/basicsr-sample_runmodel.py)，进行runmodel推理和eval评估
  - 方式2：也可使用vamc的run功能
    - 确保[basicsr-config.yaml](../vacc_code/build/basicsr-config.yaml)内dataset.path为验证集数据集路径，数据量为全部，执行以下命令后会在三件套同级目录下生成推理结果nzp目录
    ```bash
    vamc run ../vacc_code/build/basicsr-config.yaml
    ```
    - 使用[basicsr-vamp_eval.py](../vacc_code/vdsp_params/basicsr-vamp_eval.py)解析npz，绘图并统计精度（保证上面跑完全量的验证集）：
    ```bash
    python ../vacc_code/vdsp_params/basicsr-vamp_eval.py \
    --gt_dir denoising/SIDD/val/target \
    --input_npz_path datalist_npz.txt \
    --out_npz_dir deploy_weights/hinet-int8-percentile-1_256_256-debug-result \
    --input_shape 256 256 \
    --draw_dir npz_draw_result
    ```

2. vsx
   - [doc_vsx.md](../../../docs/doc_vsx.md)，参考文档安装推理`vsx`工具
   - [basicsr-vsx_inference.py](../vacc_code/vsx/basicsr-vsx_inference.py)，配置参数后执行，进行runstream推理和eval评估


### step.5 性能精度
1. 获取[vamp](../../../docs/doc_vamp.md)工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，生成`npz`格式的推理数据以及对应的`datalist_npz.txt`
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path SIDD/val/input \
    --target_dir SIDD/val/input_npz \
    --text_path datalist_npz.txt
    ```
3. 性能测试，配置vdsp参数[basicsr-hinet_sidd_1x-vdsp_params.json](../vacc_code/vdsp_params/basicsr-hinet_sidd_1x-vdsp_params.json)，执行：
    ```bash
    ./vamp -m deploy_weights/HINet-SIDD-1x-fp16-none-1_3_256_256-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/basicsr-hinet_sidd_1x-vdsp_params.json \
    -i 1 -p 1 -b 1  -s [1,256,256]
    ```
4. 精度测试
    ```bash
    ./vamp -m deploy_weights/HINet-SIDD-1x-fp16-none-1_3_256_256-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/basicsr-hinet_sidd_1x-vdsp_params.json \
    --datalist datalist_npz.txt \
    --path_output ./outputs \
    -i 1 -p 1 -b 1  -s [1,256,256]
    ```
5. 基于[basicsr-vamp_eval.py](../vacc_code/vdsp_params/basicsr-vamp_eval.py)，解析输出结果和精度评估
    ```bash
    python  ../vacc_code/vdsp_params/basicsr-vamp_eval.py \
    --gt_dir denoising/SIDD/val/target \
    --input_npz_path datalist_npz.txt \
    --out_npz_dir outputs \
    --input_shape 256 256 \
    --draw_dir output_draw_result \
    --vamp_flag
    ```

- <details><summary>性能&精度</summary>

    ```
    ./vamp_2.1.0 -m deploy_weights/HINet-SIDD-0.5x-fp16-none-1_3_256_256-vacc/mod --vdsp_params vdsp_params/basicsr-hinet_sidd_0.5x-vdsp_params.json -i 1 p 1 -b 1
    model input shape 0: [3,256,256], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 94.8539
    temperature (°C): 51.659
    card power (W): 59.7493
    die memory used (MB): 1260.86
    throughput (qps): 102.118
    e2e latency (us):
        avg latency: 24356
        min latency: 10385
        max latency: 32296
        p50 latency: 22769
        p90 latency: 29279
        p95 latency: 29308
        p99 latency: 29387
    model latency (us):
        avg latency: 24307
        min latency: 10341
        max latency: 32213
        p50 latency: 22736
        p90 latency: 29230
        p95 latency: 29255
        p99 latency: 29328

    ./vamp_2.1.0 -m deploy_weights/HINet-SIDD-0.5x-int8-percentile-1_3_256_256-vacc/mod --vdsp_params vdsp_params/basicsr-hinet_sidd_0.5x-vdsp_params.json -i 1 p 1 -b 1
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 94.72
    temperature (°C): 49.4128
    card power (W): 56.9398
    die memory used (MB): 1197.55
    throughput (qps): 356.638
    e2e latency (us):
        avg latency: 6891
        min latency: 3369
        max latency: 8843
        p50 latency: 6091
        p90 latency: 8290
        p95 latency: 8321
        p99 latency: 8364
    model latency (us):
        avg latency: 6848
        min latency: 3314
        max latency: 8782
        p50 latency: 6034
        p90 latency: 8244
        p95 latency: 8273
        p99 latency: 8315


    ./vamp_2.1.0 -m deploy_weights/HINet-SIDD-1x-fp16-none-1_3_256_256-vacc/mod --vdsp_params vdsp_params/basicsr-hinet_sidd_0.5x-vdsp_params.json -i 1 p 1 -b 1
    model input shape 0: [3,256,256], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 94.506
    temperature (°C): 57.1371
    card power (W): 61.3472
    die memory used (MB): 1516.91
    throughput (qps): 27.1975
    e2e latency (us):
        avg latency: 91738
        min latency: 37287
        max latency: 110667
        p50 latency: 73921
        p90 latency: 110294
        p95 latency: 110369
        p99 latency: 110510
    model latency (us):
        avg latency: 91689
        min latency: 37234
        max latency: 110606
        p50 latency: 73868
        p90 latency: 110245
        p95 latency: 110321
        p99 latency: 110456

    ./vamp_2.1.0 -m deploy_weights/HINet-SIDD-1x-int8-percentile-1_3_256_256-vacc/mod --vdsp_params vdsp_params/basicsr-hinet_sidd_0.5x-vdsp_params.json -i 1 p 1 -b 1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 94.9062
    temperature (°C): 56.5353
    card power (W): 67.0659
    die memory used (MB): 1293.02
    throughput (qps): 140.989
    e2e latency (us):
        avg latency: 17615
        min latency: 7715
        max latency: 21612
        p50 latency: 14638
        p90 latency: 21181
        p95 latency: 21202
        p99 latency: 21246
    model latency (us):
        avg latency: 17565
        min latency: 7664
        max latency: 21571
        p50 latency: 14566
        p90 latency: 21130
        p95 latency: 21154
        p99 latency: 21204


    ./vamp_2.1.0 -m deploy_weights/HINet-GoPro-fp16-none-1_3_256_256-vacc/mod --vdsp_params vdsp_params/basicsr-hinet_sidd_0.5x-vdsp_params.json -i 1 p 1 -b 1
    model input shape 0: [3,256,256], dtype: u1
    load model and init graph done
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 93.7987
    temperature (°C): 57.3187
    card power (W): 62.381
    die memory used (MB): 1516.87
    throughput (qps): 28.7273
    e2e latency (us):
        avg latency: 86855
        min latency: 35472
        max latency: 104689
        p50 latency: 70031
        p90 latency: 104388
        p95 latency: 104428
        p99 latency: 104526
    model latency (us):
        avg latency: 86806
        min latency: 35421
        max latency: 104633
        p50 latency: 69970
        p90 latency: 104333
        p95 latency: 104379
        p99 latency: 10447

    ./vamp_2.1.0 -m deploy_weights/HINet-GoPro-int8-percentile-1_3_256_256-vacc/mod --vdsp_params vdsp_params/basicsr-hinet_sidd_0.5x-vdsp_params.json -i 1 p 1 -b 1
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 94.7222
    temperature (°C): 53.1718
    card power (W): 51.8988
    die memory used (MB): 1292.95
    throughput (qps): 100.878
    e2e latency (us):
        avg latency: 24643
        min latency: 6951
        max latency: 36986
        p50 latency: 24463
        p90 latency: 36668
        p95 latency: 36699
        p99 latency: 36762
    model latency (us):
        avg latency: 24597
        min latency: 6942
        max latency: 36967
        p50 latency: 24416
        p90 latency: 36620
        p95 latency: 36647
        p99 latency: 3670

    # SIDD 1_3_256_256
    HINet-SIDD-0.5x_dynamic.onnx
    mean psnr: 39.67574154791744, mean ssim: 0.9198508429400674
    HINet-SIDD-0.5x-fp16-none-1_3_256_256-vacc
    mean psnr: 38.18951026987732, mean ssim: 0.9129203625641
    HINet-SIDD-0.5x-int8-percentile-1_3_256_256-vacc
    mean psnr: 37.363835573085375, mean ssim: 0.8944722407343576

    HINet-SIDD-1x_dynamic.onnx
    mean psnr: 39.5456103105865, mean ssim: 0.9207016108363085
    HINet-SIDD-1x-fp16-none-1_3_256_256-vacc
    mean psnr: 38.5875888195824, mean ssim: 0.9176884144116089
    HINet-SIDD-1x-int8-percentile-1_3_256_256-vacc
    mean psnr: 37.51703078296049, mean ssim: 0.9013895833498801


    # GoPro 1_3_256_256
    HINet-GoPro_dynamic.onnx
    mean psnr: 27.46925233341911, mean ssim: 0.8873408268442899
    HINet-GoPro-fp16-none-1_3_256_256-vacc
    mean psnr: 27.418455990682766, mean ssim: 0.8846790095232697
    HINet-GoPro-int8-percentile-1_3_256_256-vacc
    mean psnr: 27.075464575772976, mean ssim: 0.8676436905879417

    # Rain13k Test100 1_3_256_256
    HINet-Rain13k_dynamic.onnx
    mean psnr: 24.93939841449301, mean ssim: 0.8440453416259113
    HINet-Rain13k-fp16-none-1_3_256_256-vacc
    mean psnr: 26.633359780122817, mean ssim: 0.8581676094035097
    HINet-GoPro-int8-percentile-1_3_256_256-vacc
    mean psnr: 26.401614182898694, mean ssim: 0.8509198893952318

    ```
    </details>

