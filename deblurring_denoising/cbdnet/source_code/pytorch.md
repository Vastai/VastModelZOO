
### step.1 准备模型
```
link: https://github.com/IDKiro/CBDNet-pytorch
branch: master
commit: 09a2e55b2098039ee99ada8c634a06fc28c6d8a1
```
- 克隆原始仓库
- 原始模型forward返回两个值[model/cbdnet.py#L138](https://github.com/IDKiro/CBDNet-pytorch/blob/master/model/cbdnet.py#L138)，修改只返回最后一个`return noise_level, out`
- 将文件[export.py](./export.py)移动至仓库目录，配置原始模型权重路径，执行导出onnx或torchscript


### step.2 准备数据集
- 下载[SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)数据集
- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件


### step.3 模型转换
1. 获取vamc模型转换工具

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
    --gt_dir SIDD/val/target \
    --input_npz_path datalist_npz.txt \
    --out_npz_dir deploy_weights/cbdnet-fp16-none-1_3_256_256-debug-result \
    --input_shape 256 256 \
    --draw_dir npz_draw_result
    ```

2. vsx
   - [doc_vsx.md](../../../docs/doc_vsx.md)，参考文档安装推理`vsx`工具
   - [vsx_inference.py](../vacc_code/vsx/vsx_inference.py)，配置参数后执行，进行runstream推理和eval评估


### step.5 性能精度
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，生成`npz`格式的推理数据以及对应的`datalist_npz.txt`
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path SIDD/val/input \
    --target_dir SIDD/val/input_npz \
    --text_path datalist_npz.txt
    ```
3. 性能测试，配置vdsp参数[pytorch-cbdnet-vdsp_params.json](../vacc_code/vdsp_params/pytorch-cbdnet-vdsp_params.json)，执行：
    ```bash
    ./vamp -m deploy_weights/cbdnet-fp16-none-1_3_256_256-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/pytorch-cbdnet-vdsp_params.json \
    -i 1 -p 1 -b 1  -s [1,256,256]
    ```
4. 精度测试
    ```bash
    ./vamp -m deploy_weights/cbdnet-fp16-none-1_3_256_256-vacc/mod \
    --vdsp_params vacc_code/vdsp_params/pytorch-cbdnet-vdsp_params.json \
    --datalist datalist_npz.txt \
    --path_output ./outputs \
    -i 1 -p 1 -b 1  -s [1,256,256]
    ```
5. 基于[vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析输出结果和精度评估
    ```bash
    python  ../vacc_code/vdsp_params/vamp_eval.py \
    --gt_dir SIDD/val/target \
    --input_npz_path datalist_npz.txt \
    --out_npz_dir outputs \
    --input_shape 256 256 \
    --draw_dir output_draw_result \
    --vamp_flag
    ```

- <details><summary>性能&精度</summary>

    ```
    vamp -m deploy_weights/cbdnet-int8-max-1_3_256_256-vacc/mod --vdsp_params deblurring_denoising/cbdnet/vacc_code/vdsp_params/pytorch-cbdnet-vdsp_params.json -i 1 p 1 -b 1 

    [max_batch_size_]: 5
    [input_count]: 1
    input[0]: [3,256,256], dtype:u1
    [output_count]: 1
    output[0]: [1,3,256,256], dtype:f2

    *** number of instances in each device:1 ****
    Inference start time: [2024-01-04 14:05:14]
    Inference end time: [2024-01-04 14:05:15]
    number of samples in each process: 1024
    number of process:1

    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    samples: 1024
    forwad time (s): 1.31991
    throughput (qps): 775.808
    ai utilize (%): 96.4176
    die memory used (MB): 1356.75
    e2e latency (us):
        avg latency: 21697
        min latency: 1883
        max latency: 21965
    model latency (us):
        avg latency: 1241
        min latency: 1241
        max latency: 1241


    vamp -m deploy_weights/cbdnet-fp16-none-1_3_256_256-vacc/mod --vdsp_params deblurring_denoising/cbdnet/vacc_code/vdsp_params/pytorch-cbdnet-vdsp_params.json -i 1 p 1 -b 1 

    [max_batch_size_]: 2
    [input_count]: 1
    input[0]: [3,256,256], dtype:u1
    [output_count]: 1
    output[0]: [1,3,256,256], dtype:f2

    *** number of instances in each device:1 ****
    Inference start time: [2024-01-04 14:02:09]
    Inference end time: [2024-01-04 14:02:14]
    number of samples in each process: 1024
    number of process:1

    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    samples: 1024
    forwad time (s): 4.8147
    throughput (qps): 212.682
    ai utilize (%): 96.9891
    die memory used (MB): 1372.72
    e2e latency (us):
        avg latency: 79259
        min latency: 28144
        max latency: 93870
    model latency (us):
        avg latency: 4559
        min latency: 4559
        max latency: 4559

    SIDD/val/input
    checkpoint.onnx
    mean psnr: 38.26203011251678, mean ssim: 0.9028234571390887

    cbdnet-fp16-none-1_3_256_256-vacc
    mean psnr: 38.26254223499879, mean ssim: 0.9028261532777517

    cbdnet-int8-max-1_3_256_256-vacc
    mean psnr: 37.51653101431047, mean ssim: 0.8882793916856151
    ```
    </details>
