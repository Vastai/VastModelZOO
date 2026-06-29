## step.1 获取模型

1. 获取原始权重
    ```bash
    mkdir weights
    cd weights
    wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar
    wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar
    tar -xvf PP-LCNet_x0_25_textline_ori_infer.tar
    tar -xvf PP-LCNet_x1_0_textline_ori_infer.tar
    ```

1. 导出onnx

    ```bash
    Python 3.10.18
    paddleocr                 3.2.0
    paddlepaddle-gpu          3.0.0
    paddlex                   3.2.1
    paddle2onnx               2.0.0
    onnx                      1.17.0
    onnx_graphsurgeon         0.5.8
    onnxoptimizer             0.3.13
    onnxruntime               1.22.1
    onnxsim                   0.4.36
    torch                     2.8.0
    torchaudio                2.8.0
    torchvision               0.23.0

    paddle2onnx --model_dir weights/PP-LCNet_x0_25_textline_ori_infer \
                --model_filename inference.json \
                --params_filename inference.pdiparams \
                --save_file weights/PP-LCNet_x0_25_textline_ori_infer_inference.onnx \
                --opset_version 11 --enable_onnx_checker True

    paddle2onnx --model_dir weights/PP-LCNet_x1_0_textline_ori_infer \
                --model_filename inference.json \
                --params_filename inference.pdiparams \
                --save_file weights/PP-LCNet_x1_0_textline_ori_infer_inference.onnx \
                --opset_version 11 --enable_onnx_checker True

    onnxsim weights/PP-LCNet_x0_25_textline_ori_infer_inference.onnx weights/PP-LCNet_x0_25_textline_ori_infer_inference_sim.onnx
    onnxsim weights/PP-LCNet_x1_0_textline_ori_infer_inference.onnx weights/PP-LCNet_x1_0_textline_ori_infer_inference_sim.onnx
    ```

## step.2 获取数据集
- 参考：[PaddleX](https://paddlepaddle.github.io/PaddleX/3.4/module_usage/tutorials/ocr_modules/textline_orientation_classification.html#411-demo)
    ```bash
    wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/textline_orientation_example_data.tar -P ./dataset
    tar -xf ./dataset/textline_orientation_example_data.tar -C ./dataset/
    ```

## step.3 模型转换

1. 根据具体模型，修改编译配置
    - [pplcnet_textline_ori.yaml](../build_in/build/pplcnet_textline_ori.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    vamc compile ../build_in/build/pplcnet_textline_ori.yaml
    ```

## step.4 模型推理

- 参考：[vsx_infer.py](../build_in/vsx/python/vsx_infer.py)
    ```bash
    python cv/classification/pplcnet_doc_ori/build_in/vsx/python/vsx_infer.py \
    --image_dir datasets/textline_orientation_example_data \
    --label_file datasets/textline_orientation_example_data/val.txt \
    --num_images -1 \
    --model_prefix_path deploy_weights/PP-LCNet_x1_0_textline_ori_infer_inference_sim_vacc_int8_kl_divergence/mod \
    --vdsp_params_info cv/classification/pplcnet_textline_ori/build_in/vdsp_params/pplcnet_x1_0_textline_ori-vdsp_params-resize.json \
    --output_file vsx_pred.txt
    ```

- 测试结论
    - paddle官方预处理为: resize=[h80, w160]
    - 实测paddle和onnxruntime后端，在此数据集下`PP-LCNet_x0_25_textline_ori`精度略优于`PP-LCNet_x1_0_textline_ori`
    - vacc fp16精度, vacc和onnx对齐
    - vacc int8精度, 有掉点，kl_divergence量化方式精度相对更好


    | weights | backend | preprocess | metric | tips |
    | :---: | :---: | :---: | :---: | :---: |
    | [PP-LCNet_x0_25_textline_ori_infer.tar](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar) | paddleocr/paddlex | resize=[h80, w160] | Top-1: 89.50% | [paddle_eval.py](../source_code/common/paddle_eval.py) |
    | PP-LCNet_x0_25_textline_ori_infer_inference_sim.onnx | onnx/onnxruntime | resize=[h80, w160] | Top-1: 91.00% | [onnx_eval.py](../source_code/common/onnx_eval.py), baseline |
    | PP-LCNet_x0_25_textline_ori_infer_inference_vacc_fp16| vacc | resize=[h80, w160] | Top-1: **90.00%** |
    | PP-LCNet_x0_25_textline_ori_infer_inference_vacc_int8_mse| vacc | resize=[h80, w160] | Top-1: 49.00% |
    | PP-LCNet_x0_25_textline_ori_infer_inference_vacc_int8_max| vacc | resize=[h80, w160] | Top-1: 51.50% |
    | PP-LCNet_x0_25_textline_ori_infer_inference_vacc_int8_percentile| vacc | resize=[h80, w160] | Top-1: 57.50%|
    | PP-LCNet_x0_25_textline_ori_infer_inference_vacc_int8_kl_divergence| vacc | resize=[h80, w160] | Top-1: **68.00%** |


    | weights | backend | preprocess | metric | tips |
    | :---: | :---: | :---: | :---: | :---: |
    | [PP-LCNet_x1_0_textline_ori_infer.tar](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar) | paddleocr/paddlex | resize=[h80, w160] | Top-1: 87.00% | [paddle_eval.py](../source_code/common/paddle_eval.py) |
    | PP-LCNet_x1_0_textline_ori_infer_inference_sim.onnx | onnx/onnxruntime | resize=[h80, w160] | Top-1: 85.50% | [onnx_eval.py](../source_code/common/onnx_eval.py), baseline |
    | PP-LCNet_x1_0_textline_ori_infer_inference_vacc_fp16| vacc | resize=[h80, w160] | Top-1: **86.50%** |
    | PP-LCNet_x1_0_textline_ori_infer_inference_vacc_int8_mse| vacc | resize=[h80, w160] | Top-1: 77.00% |
    | PP-LCNet_x1_0_textline_ori_infer_inference_vacc_int8_max| vacc | resize=[h80, w160] | Top-1: 62.00% |
    | PP-LCNet_x1_0_textline_ori_infer_inference_vacc_int8_percentile| vacc | resize=[h80, w160] | Top-1: 73.50% |
    | PP-LCNet_x1_0_textline_ori_infer_inference_vacc_int8_kl_divergence| vacc | resize=[h80, w160] | Top-1: **84.00%** |


## step.5 性能精度测试
1. 性能测试
- VDSP配置：[pplcnet_x1_0_textline_ori-vdsp_params-resize.json](../build_in/vdsp_params/pplcnet_x1_0_textline_ori-vdsp_params-resize.json)

    ```bash
    vamp -m deploy_weights/PP-LCNet_x1_0_textline_ori_infer_inference_sim_vacc_int8_kl_divergence/mod \
    --vdsp_params cv/classification/pplcnet_textline_ori/build_in/vdsp_params/pplcnet_x1_0_textline_ori-vdsp_params-resize.json \
    -i 1 p 1 -b 1 -s [3,80,160]
    ```
    > 模型较小，在不同硬件下，考虑加大batchsize获取最佳吞吐

2. 精度测试
- 同上参考：[vsx_infer.py](../build_in/vsx/python/vsx_infer.py)


## Appendix
- Paddle和ONNX测评对比
    - [paddle_eval.py](../source_code/common/paddle_eval.py)
    - [onnx_eval.py](../source_code/common/onnx_eval.py)
- [PaddleX官方描述](https://paddlepaddle.github.io/PaddleX/3.4/module_usage/tutorials/ocr_modules/textline_orientation_classification.html#41)
    - 精度指标
        - PP-LCNet_x0_25_textline_ori, Top-1 Acc 98.85%
        - PP-LCNet_x1_0_textline_ori, Top-1 Acc 99.42%
    - 应该不是上文中测试集测出来的精度指标；此数据集实测Paddle原始推理权重及PaddleOCR后端指标依次为：TOP-1 89.50%和87.00%