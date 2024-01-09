### step.1 获取预训练模型

目前Compiler暂不支持官方提供的转换脚本生成的`onnx`以及`torchscript`模型生成三件套，需基于以下脚本生成`torchscript`格式进行三件套转换(onnx暂不支持)

```python
import os
import torch
from ultralytics import YOLO

models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
size = [416, 608, 640, 1024, 2048]

for m in models:
    for s in size:
        MODEL = os.path.join('./weights', m)
        model = YOLO(MODEL)
        model.to("cpu")

        img_tensor=torch.zeros(1, 3, s, s)
        scripted_model = torch.jit.trace(model.model, img_tensor, check_trace=False).eval()

        torch.jit.save(scripted_model, os.path.join('./torchscript', m.split('.')[0] + '-' + str(s) + '.torchscript.pt'))
```

### step.2 准备数据集
- 准备[COCO](https://cocodataset.org/#download)数据集


### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[config_ultralytics.yaml](../vacc_code/build/config_ultralytics.yaml)：
    ```bash
    vamc build ../vacc_code/build/config_ultralytics.yaml
    ```

### step.4 性能精度
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
3. 性能测试
    ```bash
    vamp -m deploy_weights/yolo8s-int8-percentile-3_640_640-vacc/yolo8s --vdsp_params ../vacc_code/vdsp_params/ultralytics-yolo8s-vdsp_params.json -i 1 p 1 -b 1
    ```

    ```bash
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 95.5
    temperature (°C): 41.2264
    card power (W): 36.1655
    die memory used (MB): 1357.08
    throughput (qps): 625.875
    e2e latency (us):
        avg latency: 3879
        min latency: 2384
        max latency: 5233
        p50 latency: 3824
        p90 latency: 4685
        p95 latency: 4719
        p99 latency: 4779
    model latency (us):
        avg latency: 3842
        min latency: 2349
        max latency: 5218
        p50 latency: 3772
        p90 latency: 4646
        p95 latency: 4674
        p99 latency: 4737
    
    ```

4. npz结果输出
    ```bash
    vamp -m deploy_weights/yolo8s-int8-percentile-3_640_640-vacc/yolo8s --vdsp_params ../vacc_code/vdsp_params/ultralytics-yolo8s-vdsp_params.json -i 1 p 1 -b 1 --datalist datasets/coco_npz_datalist.txt --path_output npz_output
    ```
5. [vamp_decode.py](../vacc_code/vdsp_params/vamp_decode.py)，解析vamp输出的npz文件，进行绘图和保存txt结果
    ```bash
    python ../vacc_code/vdsp_params/vamp_decode.py --txt result_npz --label_txt datasets/coco.txt --input_image_dir datasets/coco_val2017 --model_size 640 640 --vamp_datalist_path datasets/coco_npz_datalist.txt --vamp_output_dir npz_output
    ```
6. [eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt path/to/vamp_draw_output
   ```
