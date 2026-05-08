
# PPOCRv5 REC

- [PP-OCRv5 介绍](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.5.0/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.md)

## Build_In Deploy

```
link: https://github.com/PaddlePaddle/PaddleOCR
tag: v3.5.0
```

### step.1 获取推理模型
1. 首先，需要进入到PaddleOCR工程主目录，安装PaddleOCR：
```
cd PaddleOCR
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. 下载推理模型

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar

mkdir -p weights/
tar -xvf PP-OCRv5_mobile_rec_infer.tar -C weights/
tar -xvf PP-OCRv5_server_rec_infer.tar -C weights/
```

3. 将推理模型转换为onnx

```bash
pip install onnx==1.14.0 onnxsim==0.4.35 onnxruntime==1.13.1 paddle2onnx==2.0.0

# paddle静态模型导出onnx
paddle2onnx --model_dir weights/PP-OCRv5_mobile_rec_infer \
            --model_filename inference.json \
            --params_filename inference.pdiparams \
            --save_file weights/PP-OCRv5_mobile_rec_infer_inference.onnx \
            --opset_version 11 --enable_onnx_checker True
# onnx简化
onnxsim weights/PP-OCRv5_mobile_rec_infer_inference.onnx weights/PP-OCRv5_mobile_rec_infer_inference_sim.onnx

paddle2onnx --model_dir weights/PP-OCRv5_server_rec_infer \
            --model_filename inference.json \
            --params_filename inference.pdiparams \
            --save_file weights/PP-OCRv5_server_rec_infer_inference.onnx \
            --opset_version 11 --enable_onnx_checker True
onnxsim weights/PP-OCRv5_server_rec_infer_inference.onnx weights/PP-OCRv5_server_rec_infer_inference_sim.onnx
```

4. onnx算子替换

> 为支持vacc编译，server模型需要将autopad转换至pad算子

- 参考：[convert_autopad_to_pads.py](../../text_detection/dbnet/source_code/ppocr_v5/convert_autopad_to_pads.py)

```bash
python cv/text_detection/dbnet/source_code/ppocr_v5/convert_autopad_to_pads.py \
--input_onnx weights/PP-OCRv5_server_rec_infer_inference_sim.onnx \
--output_onnx weights/PP-OCRv5_server_rec_infer_inference_sim_pads.onnx
```

### step.2 准备数据集
- 文本检测数据集：[SWHL/text_rec_test_dataset](https://hf-mirror.com/datasets/SWHL/text_rec_test_dataset)


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [ppocr_v5_rec.yaml](./build_in/build/ppocr_v5_rec.yaml)
        
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd ppocr_v5_rec
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/ppocr_v5_rec.yaml
    ```

### step.4 精度测试
> 此精度评估脚本修改自仓库：[HoVDuc/ppocrv5-onnx](https://github.com/HoVDuc/ppocrv5-onnx)

1. 参考：[rec_eval.py](../../text_detection/dbnet/source_code/ppocr_v5/rec_eval.py)，进行onnx或vacc模型的精度评估
    ```bash
    # 使用固定尺寸推理，use_fixed_shape=True
    # 自动下载huggingface数据集
    export HF_ENDPOINT=https://hf-mirror.com
    python cv/text_detection/dbnet/source_code/ppocr_v5/rec_eval.py
    ```

    ```bash
    # PP-OCRv5_mobile_rec_infer_inference_sim.onnx
    use_fixed_shape=False
    {'ExactMatch': 0.729, 'CharMatch': 0.9123}
    use_fixed_shape=True
    {'ExactMatch': 0.6903, 'CharMatch': 0.8546}

    # PP-OCRv5_mobile_rec_infer_fp16_320_runtream
    use_fixed_shape=True
    {'ExactMatch': 0.7258, 'CharMatch': 0.8687}

    # PP-OCRv5_server_rec_infer_inference_sim.onnx
    use_fixed_shape=False
    {'ExactMatch': 0.8032, 'CharMatch': 0.9376}
    use_fixed_shape=True
    {'ExactMatch': 0.7548, 'CharMatch': 0.8759}

    # PP-OCRv5_server_rec_infer_fp16_960_runstream
    use_fixed_shape=True
    {'ExactMatch': 0.7871, 'CharMatch': 0.8979}
    ```

### step.5 性能测试
1. 配置VDSP参数：[ppocr-ch_PP_OCRv5_rec-vdsp_params.json](./build_in/vdsp_params/ppocr-ch_PP_OCRv5_rec-vdsp_params.json)

2. 执行VAMP性能测试
    ```bash
    ./vamp -m deploy_weights/PP-OCRv5_mobile_rec_infer_fp16_960/mod \
    --vdsp_params cv/text_detection/dbnet/build_in/vdsp_params/ppocr-ch_PP_OCRv5_rec-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,48,320]
    ```