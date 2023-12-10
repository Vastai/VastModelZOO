### step.1 获取预训练模型

ppdet库可以利用paddle2onnx工具将原始模型转为onnx格式，转换前需要修改源码去除后处理，主要是在[yolo.py](https://github.com/PaddlePaddle/PaddleDetection/blob/release%2F2.6/ppdet/modeling/architectures/yolo.py#L144)中，将return output改为return yolo_head_outs即可

如果原模型是pdparams格式，则需要先转换为inference格式，具体步骤如下

```bash
cd PaddleDetection

python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml -o weights=models/yolov3/yolov3_darknet53_270e_coco.pdparams --output_dir inference_model

paddle2onnx --model_dir inference_model/yolov3_darknet53_270e_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 11 --save_file yolov3_darknet53.onnx

```

如果需要修改输入shape，则可以按照官方说明指定shape，或者在配置文件中进行修改

### step.2 准备数据集
- 准备[COCO](https://cocodataset.org/#download)数据集

### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[config_ppdet.yaml](../vacc_code/build/config_ppdet.yaml)：
    ```bash
    vamc build ../vacc_code/build/config_ppdet.yaml
    ```

### step.4 性能精度
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
3. 性能测试
    ```bash
    vamp -m deploy_weights_ppdet/yolov3-int8-kl_divergence-3_608_608-vacc/yolov3 --vdsp_params ../vacc_code/vdsp_params/ppdet-yolov3_darknet53-vdsp_params.json -i 2 p 2 -b 1
    ```
4. npz结果输出
    ```bash
    vamp -m deploy_weights_ppdet/yolov3-int8-kl_divergence-3_608_608-vacc/yolov3 --vdsp_params ../vacc_code/vdsp_params/ppdet-yolov3_darknet53-vdsp_params.json -i 2 p 2 -b 1 --datalist datasets/coco_npz_datalist.txt --path_output npz_output
    ```
5. [vamp_decode_ppdet.py](../vacc_code/vdsp_params/vamp_decode_ppdet.py)，解析vamp输出的npz文件，进行绘图和保存txt结果
    ```bash
    python ../vacc_code/vdsp_params/vamp_decode_ppdet.py --txt result_npz_txt --label_txt datasets/coco.txt --input_image_dir datasets/coco_val2017 --model_size 608 608 --vamp_datalist_path datasets/coco_npz_datalist.txt --vamp_output_dir npz_output
    ```
6. [eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt path/to/vamp_draw_output
   ```
