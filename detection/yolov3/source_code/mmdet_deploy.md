### step.1 获取预训练模型

mmdet提供了将原始torch模型转为onnx格式的脚本，运行以下命令即可

```bash
python tools/deployment/pytorch2onnx.py configs/yolo/yolov3_mobilenetv2_mstrain-416_300e_coco.py models/yolov3/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth  --output-file models/yolov3/yolov3_mobilenetv2-416.onnx --shape 416 416 --verify --skip-postprocess

```

### step.2 准备数据集
- 准备[COCO](https://cocodataset.org/#download)数据集


### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[config_mmdet.yaml](../vacc_code/build/config_mmdet.yaml)：
    ```bash
    vamc build ../vacc_code/build/config_mmdet.yaml
    ```

### step.4 性能精度
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
3. 性能测试
    ```bash
    vamp -m deploy_weights/yolov3-int8-kl_divergence-3_320_320-vacc/yolov3 --vdsp_params ../vacc_code/vdsp_params/mmdet-yolov3-vdsp_params.json -i 2 p 2 -b 1
    ```
4. npz结果输出
    ```bash
    vamp -m deploy_weights/yolov3-int8-kl_divergence-3_320_320-vacc/yolov3 --vdsp_params ../vacc_code/vdsp_params/mmdet-yolov3-vdsp_params.json -i 2 p 2 -b 1 --datalist datasets/coco_npz_datalist.txt --path_output npz_output
    ```
5. [vamp_decode.py](../vacc_code/vdsp_params/vamp_decode.py)，解析vamp输出的npz文件，进行绘图和保存txt结果
    ```bash
    python ../vacc_code/vdsp_params/vamp_decode.py --txt result_npz --label_txt datasets/coco.txt --input_image_dir datasets/coco_val2017 --model_size 320 320 --vamp_datalist_path datasets/coco_npz_datalist.txt --vamp_output_dir npz_output
    ```
6. [eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt path/to/vamp_draw_output
   ```
