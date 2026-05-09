## PPOCRv5 DET

```
link: https://github.com/PaddlePaddle/PaddleOCR
tag: v3.5.0
```
- [PP-OCRv5介绍](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.5.0/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.md)

### step.1 模型准备
1. 首先，需要进入到PaddleOCR工程主目录，安装PaddleOCR：

```
cd PaddleOCR
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. 下载推理模型

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar

mkdir -p weights/
tar -xvf PP-OCRv5_mobile_det_infer.tar -C weights/
tar -xvf PP-OCRv5_server_det_infer.tar -C weights/
```

3. 将推理模型转换为onnx

```bash
pip install onnx==1.14.0 onnxsim==0.4.35 onnxruntime==1.13.1 paddle2onnx==2.0.0

# paddle静态模型导出onnx
paddle2onnx --model_dir weights/PP-OCRv5_mobile_det_infer \
            --model_filename inference.json \
            --params_filename inference.pdiparams \
            --save_file weights/PP-OCRv5_mobile_det_infer_inference.onnx \
            --opset_version 11 --enable_onnx_checker True
# onnx简化
onnxsim weights/PP-OCRv5_mobile_det_infer_inference.onnx weights/PP-OCRv5_mobile_det_infer_inference_sim.onnx

paddle2onnx --model_dir weights/PP-OCRv5_server_det_infer \
            --model_filename inference.json \
            --params_filename inference.pdiparams \
            --save_file weights/PP-OCRv5_server_det_infer_inference.onnx \
            --opset_version 11 --enable_onnx_checker True
onnxsim weights/PP-OCRv5_server_det_infer_inference.onnx weights/PP-OCRv5_server_det_infer_inference_sim.onnx
```

4. onnx算子替换

> 为支持vacc编译，server模型需要将autopad转换至pad算子

- 参考：[convert_autopad_to_pads.py](./ppocr_v5/convert_autopad_to_pads.py)

```bash
python cv/text_detection/dbnet/source_code/ppocr_v5/convert_autopad_to_pads.py \
--input_onnx weights/PP-OCRv5_server_det_infer_inference_sim.onnx \
--output_onnx weights/PP-OCRv5_server_det_infer_inference_sim_pads.onnx
```

### step.2 准备数据集
- 文本检测数据集：[SWHL/text_det_test_dataset](https://hf-mirror.com/datasets/SWHL/text_det_test_dataset)

### step.3 模型转换
1. 根据具体模型修改配置文件
    - [ppocr_v5_dbnet.yaml](../build_in/build/ppocr_v5_dbnet.yaml)

    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    cd dbnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/ppocr_v5_dbnet.yaml
    ```
    - 转换后将在当前目录下生成`deploy_weights`文件夹，其中包含转换后的模型文件。

### step.4 精度测试
> 此精度评估脚本修改自仓库：[HoVDuc/ppocrv5-onnx](https://github.com/HoVDuc/ppocrv5-onnx)

1. 参考：[det_eval.py](../source_code/ppocr_v5/det_eval.py)，进行onnx或vacc模型的精度评估
    ```bash
    # 使用固定尺寸推理，use_fixed_shape=True
    # 自动下载huggingface数据集
    export HF_ENDPOINT=https://hf-mirror.com
    python cv/text_detection/dbnet/source_code/ppocr_v5/det_eval.py
    ```

    ```bash
    # PP-OCRv5_mobile_det_infer_inference_sim.onnx
    use_fixed_shape=False
    {'precision': 0.7915, 'recall': 0.8266, 'hmean': 0.8087}
    use_fixed_shape=True
    {'precision': 0.7504, 'recall': 0.7922, 'hmean': 0.7707}

    # PP-OCRv5_mobile_det_infer_fp16_960_runstream
    use_fixed_shape=True
    {'precision': 0.7545, 'recall': 0.7995, 'hmean': 0.7763}


    # PP-OCRv5_server_det_infer_inference_sim.onnx
    use_fixed_shape=False
    {'precision': 0.8293, 'recall': 0.8667, 'hmean': 0.8476}
    use_fixed_shape=True
    {'precision': 0.8105, 'recall': 0.8346, 'hmean': 0.8224}

    # PP-OCRv5_server_det_infer_fp16_960_runstream
    use_fixed_shape=True
    {'precision': 0.8086, 'recall': 0.8231, 'hmean': 0.8158}
    ```

### step.5 性能测试
1. 配置VDSP参数：[ppocr-ch_PP_OCRv5_det-vdsp_params.json](../build_in/vdsp_params/ppocr-ch_PP_OCRv5_det-vdsp_params.json)

2. 执行VAMP性能测试
    ```bash
    ./vamp -m deploy_weights/PP-OCRv5_mobile_det_infer_fp16_960/mod \
    --vdsp_params cv/text_detection/dbnet/build_in/vdsp_params/ppocr-ch_PP_OCRv5_det-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,960,960]
    ```
