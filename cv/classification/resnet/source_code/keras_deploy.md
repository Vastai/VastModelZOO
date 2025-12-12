
### step.1 获取模型
⚠️ keras h5 is directly supported formats!

### step.2 获取数据集
- [校准数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [评估数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [label_list](../../common/label/imagenet.txt)
- [label_dict](../../common/label/imagenet1000_clsid_to_human.txt)

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [keras_resnet.yaml](../build_in/build/keras_resnet.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd resnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/keras_resnet.yaml
    ```

    **Note:** 需基于Python3.8环境进行模型转换操作

### step.4 模型推理

- 参考：[classification.py](../../common/vsx/classification.py)
    ```bash
    python ../../common/vsx/classification.py \
        --infer_mode sync \
        --file_path path/to/ILSVRC2012_img_val \
        --model_prefix_path deploy_weights/keras_resnet50_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/keras-resnet50-vdsp_params.json \
        --label_txt path/to/imagenet.txt \
        --save_dir ./infer_output \
        --save_result_txt result.txt \
        --device 0
    ```

- 精度评估
    ```
    python ../../common/eval/eval_topk.py ./infer_output/result.txt
    ```

    ```
    # fp16
    [VACC]:  top1_rate: 72.334 top5_rate: 90.708

    # int8
    [VACC]:  top1_rate: 72.286 top5_rate: 90.622
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置[keras-resnet50-vdsp_params.json](../build_in/vdsp_params/keras-resnet50-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/keras_resnet50_int8/mod --vdsp_params ../build_in/vdsp_params/keras-resnet50-vdsp_params.json  -i 8 -p 1 -b 2 -s [3,224,224]
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；
    
    - 数据准备，生成推理数据`npz`以及对应的`dataset.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path ILSVRC2012_img_val --target_path  input_npz  --text_path imagenet_npz.txt
    ```

    - vamp推理获取npz文件
    ```
    vamp -m deploy_weights/keras_resnet50_int8/mod --vdsp_params ../build_in/vdsp_params/keras-resnet50-vdsp_params.json  -i 8 -p 1 -b 22 -s [3,224,224] --datalist imagenet_npz.txt --path_output output
    ```

    - 解析输出结果用于精度评估，参考：[vamp_npz_decode.py](../../common/eval/vamp_npz_decode.py)
    ```bash
    python  ../../common/eval/vamp_npz_decode.py imagenet_npz.txt output imagenet_result.txt imagenet.txt
    ```
    
    - 精度评估，参考：[eval_topk.py](../../common/eval/eval_topk.py)
    ```bash
    python ../../common/eval/eval_topk.py imagenet_result.txt
    ```

### Tips
- 在keras来源模型中，有v2版本参数不太一样，注意修改[resnet_keras.yaml](../build_in/build/keras_resnet.yaml) 
  - resnet50、resnet101、resnet152所需的输入尺寸为224，input_name应设置为`input_1`，且该版本模型输入为BGR格式;
  - resnet50v2、resnet101v2、resnet152v2所需的输入尺寸为299，input_name应设置为`input_4`；
  - 同时，也可使用函数脚本式预处理[keras_preprocess.py](../build_in/build/keras_preprocess.py)，注意，v2模型应采用下述代码中的`get_image_data_v2`函数
    <details><summary>keras.yaml</summary>

    ```yaml
    name: resnet50

    frontend:
        shape:
            input_1: [1, 3, 224, 224]
        type: keras
        checkpoint: weights/keras/resnet50.h5
        dtype: fp32

    graph:
        extra_ops:
        type: null

    backend:
        type: tvm_vacc
        dtype: int8
        quantize:
            calibrate_mode: percentile
            quantize_per_channel: true

    dataset:
        type: tvm
        path: eval/ILSVRC2012_img_calib
        sampler:
            suffix: JPEG
            get_data_num: 1000
        process_ops:
            - type: CustomFunc
            module_path: classification/resnet/build_in/build/keras_preprocess.py
            func_name: get_image_data
            input_shape: [1, 3, 224, 224]

    workspace:
        work_dir: ./deploy_weights/
        save_log: true
    ```
    </details>
