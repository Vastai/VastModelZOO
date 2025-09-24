### step.1 获取预训练模型
1. 基于[yolox](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0),转换模型至torchscript格式可运行如下脚本
    ```
    python tools/export_torchscript.py -n yolox_s -c /path/to/yolox.pth --output-name /path/to/save_model.torchscript.pt
    ```
    - `n`参数对应yolox的模型名字，可选值包括['yolox_s', 'yolox_m', 'yolox_l', 'yolox_x', 'yolov3', 'yolox_tiny', 'yolox_nano']
    - `c`参数即要进行转换的模型路径
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
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)

### step.3 模型转换
1. 根据具体模型修改模型转换配置文件
    - [official_yolox.yaml](../build_in/build/official_yolox.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`

2. 模型编译    
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    cd yolox
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_yolox.yaml
    ```
    - 转换后将在当前目录下生成`deploy_weights/official_yolox_run_stream_int8`文件夹，其中包含转换后的模型文件。

### step.4 模型推理
1. 可以利用[脚本](../../common/vsx/detection.py)生成预测的txt结果

    ```
    python ../../common/vsx/detection.py \
        --file_path path/to/coco_val2017 \
        --model_prefix_path deploy_weights/official_yolox_run_stream_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-yolox_s-vdsp_params.json \
        --label_txt ../../common/label/coco.txt \
        --save_dir ./runstream_output \
        --device 0
    ```
    - 注意替换命令行中--file_path和--label_txt为实际路径

2. [eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
    ```bash
        python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt ./runstream_output
    ```
    - 测试精度数据如下：
    ```
    DONE (t=4.47s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.386
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.581
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.415
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.215
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.431
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.520
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.317
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.494
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.318
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.568
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.674
    {'bbox_mAP': 0.386, 'bbox_mAP_50': 0.581, 'bbox_mAP_75': 0.415, 'bbox_mAP_s': 0.215, 'bbox_mAP_m': 0.431, 'bbox_mAP_l': 0.52, 'bbox_mAP_copypaste': '0.386 0.581 0.415 0.215 0.431 0.520'}
    ```

### step.5 性能精度测试
1. 性能测试
    ```bash
    vamp -m deploy_weights/official_yolox_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/official-yolox_s-vdsp_params.json -i 2 p 2 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path path/to/coco_val2017 \
        --target_path  path/to/coco_val2017_npz  \
        --text_path npz_datalist.txt
    ```
    - vamp推理，获取npz结果输出
    ```bash
    vamp -m deploy_weights/official_yolox_run_stream_int8/mod \
        ../build_in/vdsp_params/official-yolox_s-vdsp_params.json \
        -i 2 p 2 -b 1\
        --datalist path/to/npz_datalist.txt \
        --path_output npz_output
    ```
    - 解析npz文件，参考：[npz_decode.py](../../common/utils/npz_decode.py)
        ```bash
        python ../../common/utils/npz_decode.py \
            --txt result_npz --label_txt ../../common/label/coco.txt \
            --input_image_dir path/to/coco_val2017 \
            --model_size 640 640 \
            --vamp_datalist_path path/to/npz_datalist.txt \
            --vamp_output_dir npz_output
        ```
    
    - 参考：[eval_map.py](../../common/eval/eval_map.py)，精度统计
    ```bash
    python ../../common/eval/eval_map.py \
        --gt path/to/instances_val2017.json \
        --txt path/to/result_npz
    ```

