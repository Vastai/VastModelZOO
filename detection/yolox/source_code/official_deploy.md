### step.1 获取预训练模型

1. 基于[yolox](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0),转换模型至torchscript格式可运行如下脚本
    ```
    python tools/export_torchscript.py -n yolox_s -c /path/to/yolox.pth -s $SIZE --output-name /path/to/save_model.torchscript.pt
    ```
    - `n`参数对应yolox的模型名字，可选值包括['yolox_s', 'yolox_m', 'yolox_l', 'yolox_x', 'yolov3', 'yolox_tiny', 'yolox_nano']
    - `c`参数即要进行转换的模型路径
    - `s`参数表示模型输入尺寸，建议32的倍数
    - `output-name`参数表示转换的torchscript模型保存路径

2. 转换模型至onnx格式可运行如下脚本
    ```
    python tools/export_onnx.py --input input -n yolox_s -c /path/to/yolox.pth -s $SIZE --output-name /path/to/save_model.onnx
    ```
    - `n`参数对应yolox的模型名字，可选值包括['yolox_s', 'yolox_m', 'yolox_l', 'yolox_x', 'yolov3', 'yolox_tiny', 'yolox_nano']
    - `c`参数即要进行转换的模型路径
    - `s`参数表示模型输入尺寸，建议32的倍数
    - `output-name`参数表示转换的onnx模型保存路径

### step.2 准备数据集
- 准备[COCO](https://cocodataset.org/#download)数据集

### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[official_yolox.yaml](../vacc_code/build/official_yolox.yaml)：
    ```bash
    vamc build ../vacc_code/build/official_yolox.yaml
    ```


### step.4 性能精度
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
3. 性能测试
    > vdsp_params中的color_cvt_code需要设置为COLOR_BGR2RGB，COLOR_NO_CHANGE会掉三到四个点
    ```bash
    vamp -m deploy_weights/yolox_s-int8-kl_divergence-3_640_640-vacc/yolox_s --vdsp_params ../vacc_code/vdsp_params/official-yolox_s-vdsp_params.json -i 2 p 2 -b 1
    ```
4. npz结果输出
    ```bash
    vamp -m deploy_weights/yolox_s-int8-kl_divergence-3_640_640-vacc/yolox_s --vdsp_params ../vacc_code/vdsp_params/official-yolox_s-vdsp_params.json -i 2 p 2 -b 1 --datalist datasets/coco_npz_datalist.txt --path_output npz_output
    ```
5. [vamp_decode.py](../vacc_code/vdsp_params/vamp_decode.py)，解析vamp输出的npz文件，进行绘图和保存txt结果
    ```bash
    python ../vacc_code/vdsp_params/vamp_decode.py --vamp_datalist_path npz_datalist.txt --label_txt datasets/coco.txt --input_image_dir datasets/coco_val2017 --vdsp_params_path yoloxs_vdsp.json  --vamp_output_dir npz_output
    ```
6. [eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt path/to/vamp_draw_output
   ```

