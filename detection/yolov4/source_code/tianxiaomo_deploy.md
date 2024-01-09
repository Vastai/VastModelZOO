
### step.1 获取预训练模型

```
link: https://github.com/Tianxiaomo/pytorch-YOLOv4
branch: master
commit: a65d219f9066bae4e12003bd7cdc04531860c672
```

- 端到端，基于源仓库脚本：[demo_pytorch2onnx.py](https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/demo_pytorch2onnx.py)，执行以下命令，转换为batch维度是动态的onnx：
    ```bash
    python demo_pytorch2onnx.py weights/yolov4.pth data/dog.jpg -1 80 416 416
    ```

### step.2 准备数据集
- 准备[COCO](https://cocodataset.org/#download)数据集


### step.3 模型转换

1. 获取[vamc](../../../docs/doc_vamc.md)模型转换工具

2. 根据具体模型修改模型转换配置文件[tianxiaomo_config.yaml](../vacc_code/build/tianxiaomo_config.yaml)：
    ```bash
    vamc build ../vacc_code/build/tianxiaomo_config.yaml
    ```

### step.4 性能精度
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
3. 性能测试
    ```bash
    ./vamp -m deploy_weights/yolov4-int8-kl_divergence-3_416_416-vacc/yolov4 --vdsp_params path/to/tianxiaomo-yolov4-vdsp_params.json -i 2 p 2 -b 1 -s [3,416,416]
    ```
4. 精度测试
    ```bash
    ./vamp -m deploy_weights/yolov4-int8-kl_divergence-3_416_416-vacc/yolov4 --vdsp_params path/to/tianxiaomo-yolov4-vdsp_params.json -i 2 p 2 -b 1 -s [3,416,416] --datalist path/to/npz_datalist.txt --path_output path/to/yolov4_vamp_output
    ```
5. [tianxiaomo_vamp_decode.py](../vacc_code/vdsp_params/tianxiaomo_vamp_decode.py)，解析vamp输出的npz文件，进行绘图和保存txt结果
    ```bash
    python ../vacc_code/vdsp_params/tianxiaomo_vamp_decode.py --input_image_dir path/to/coco_val2017 --vamp_datalist_path path/to/npz_datalist.txt --vamp_output_dir path/to/yolov4_vamp_output --vdsp_params_path path/to/tianxiaomo-yolov4-vdsp_params.json --label_txt path/to/coco.txt --draw_image True --save_dir path/to/vamp_draw_output
    ```
6. [eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt path/to/vamp_draw_output
   ```