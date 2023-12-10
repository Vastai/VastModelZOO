### step.1 获取预训练模型
```

link: https://github.com/bubbliiiing/yolov4-tiny-pytorch
branch: master
commit: 60598a259cfe64b4b87f064fbd4b183a4cdd6cba

link: https://github.com/bubbliiiing/yolov4-pytorch
branch: master
commit: b7c2212250037c262282bac06fcdfe97ac86c055
```
- 克隆原始yolov4仓库或yolov4仓库，将所有文件移动至[yolov4/source_code/bubbliiiing](../source_code/bubbliiiing)文件夹下
- 修改[yolo.py#L20](https://github.com/bubbliiiing/yolov4-pytorch/blob/master/yolo.py#L20)，yolo类中的`_defaults`参数（配置文件路径，开启letterbox，关闭cuda）
- 基于[onnx_convert.py](../source_code/bubbliiiing/onnx_convert.py)，导出onnx


### step.2 准备数据集
- 准备[COCO](https://cocodataset.org/#download)数据集


### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[bubbliiiing_config.yaml](../vacc_code/build/bubbliiiing_config.yaml)：
    ```bash
    vamc build ../vacc_code/build/bubbliiiing_config.yaml
    ```
> Tips:
> 
> bubbliiiing来源的yolov4和yolov4_tiny均只支持`forward`推理，配置表的`add_extra_ops_to_graph`参数设置为`type: null`
>

### step.4 性能精度
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
3. 性能测试
    ```bash
    ./vamp -m deploy_weights/yolov4-int8-kl_divergence-3_416_416-vacc/yolov4 --vdsp_params path/to/bubbliiiing-yolov4-vdsp_params.json -i 2 p 2 -b 1 -s [3,416,416]
    ```
4. 精度测试
    ```bash
    ./vamp -m deploy_weights/yolov4-int8-kl_divergence-3_416_416-vacc/yolov4 --vdsp_params path/to/bubbliiiing-yolov4-vdsp_params.json -i 2 p 2 -b 1 -s [3,416,416] --datalist path/to/npz_datalist.txt --path_output path/to/yolov4_vamp_output
    ```
5. [bubbliiiing_vamp_decode.py](../vacc_code/vdsp_params/bubbliiiing_vamp_decode.py)，解析vamp输出的npz文件，进行绘图和保存txt结果
    ```bash
    python ../vacc_code/vdsp_params/bubbliiiing_vamp_decode.py --input_image_dir path/to/coco_val2017 --vamp_datalist_path path/to/npz_datalist.txt --vamp_output_dir path/to/yolov4_vamp_output --vdsp_params_path path/to/bubbliiiing-yolov4-vdsp_params.json --label_txt path/to/coco.txt --draw_image True --save_dir path/to/vamp_draw_output
    ```
6. [eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt path/to/vamp_draw_output
   ```
