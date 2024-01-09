# pytorch

### step.1 获取预训练模型

1. 修改原始库中`retinanet/model.py`，替换为该目录下`pytorch/model.py`文件

2. 运行如下脚本进行模型转换，生成onnx格式

    ```python
    import torch
    from retinanet.model import resnet50

    model_path = 'weights/coco_resnet_50_map_0_335_state_dict.pt'
    model = resnet50(num_classes=80,)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.training = False
    model.eval()

    input_shape = (1, 3, 1024, 1024)
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data)
    torch.jit.save(scripted_model, 'weights/retinanet.torchscript.pt')

    input_names = ["input"]

    torch_out = torch.onnx._export(model, input_data, 'weights/retinanet.onnx', export_params=True, verbose=False,
                                input_names=input_names, opset_version=11)
    ```

### step.2 准备数据集
- 准备[COCO](https://cocodataset.org/#download)数据集

### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[pytorch_config.yaml](../vacc_code/build/pytorch_config.yaml)：
    ```bash
    vamc build ../vacc_code/build/pytorch_config.yaml
    ```

### step.4 性能精度
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
3. 性能测试
    ```bash
    vamp -m deploy_weights/retinanet-int8-max-1_3_1024_1024-debug/retinanet --vdsp_params ../vacc_code/vdsp_params/pytorch-retinanet_resnet50-vdsp_params.json -i 1 -b 1 -d 0 -p 1
    ```

    ```bash
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 40.0225
    temperature (°C): 39.5858
    card power (W): 30.3704
    die memory used (MB): 1660.93
    throughput (qps): 41.5769
    e2e latency (us):
        avg latency: 52049
        min latency: 46867
        max latency: 65637
        p50 latency: 53599
        p90 latency: 56406
        p95 latency: 56446
        p99 latency: 56540
    model latency (us):
        avg latency: 51990
        min latency: 46812
        max latency: 65562
        p50 latency: 53528
        p90 latency: 56347
        p95 latency: 56391
        p99 latency: 56487
    ```
4. npz结果输出
    ```bash
    vamp -m deploy_weights/retinanet-int8-max-1_3_1024_1024-debug/retinanet --vdsp_params ../vacc_code/vdsp_params/pytorch-retinanet_resnet50-vdsp_params.json  -i 1 -b 1 -d 0 -p 1 --datalist datasets/coco_npz_datalist.txt --path_output npz_output
    ```