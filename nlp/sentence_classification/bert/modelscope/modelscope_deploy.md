## modelscope deploy
> 模型指定为 `damo/nlp_structbert_sentiment-classification_chinese-base`

### step.1 获取模型
- 首先执行[export.py](./source_code/export.py )通过 modelscope 在线下载模型，并导出onnx。注意，以下所有操作的python依赖可参考`appending`
    ```bash
    python source_code/export.py
    ```

### step.2 获取数据集
- [评估数据集](https://drive.google.com/drive/folders/1HREE-mJNBKkKQPuXU7gLNXBYFvV8ZEZ2)
- [校准数据集](https://drive.google.com/drive/folders/1Zi_LY-EHDVT3cGo9Uh1ArOwBzGSJW5xo)
- [jd_label.txt](https://drive.google.com/drive/folders/14zKjAwdLy_khn8R6jQkBop_V7mmi57Xl)
- tvm在1.5之后的版本对nlp系列模型的输入有特殊的要求， 所以需要将tokenizer之后的数据进行相应的处理。用户可直接下载评估数据集和校准数据集， 也可以通过[gen_dataset.py](./source_code/gen_dataset.py)制作数据集
    ```bash
    python source_code/gen_dataset.py
    ```

### step.3 模型转换
1. 根据具体模型修改配置文件
    - [bert_modelscope.yaml](./build_in/build/bert_modelscope.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    vamc build ./build_in/build/bert_modelscope.yaml
    ```

### step.4 模型推理
> 基于vsx进行推理
- 运行 [vsx_infer.py](./build_in/vsx/vsx_infer.py) 进行推理和精度评测
    ```
    python ./build_in/vsx/vsx_infer.py --model_prefix bert/mod \
        --vdsp_json ./build_in/vsx/vdsp_params.json \
        --data_dir npz_files \
        --label jd_label.txt
    ```

### appending

- python requirements

    ```bash
    onnx                     1.14.0
    onnxruntime              1.13.1
    protobuf                 3.20.3
    torch                    2.0.1
    transformers             4.31.0
    modelscope               1.13.3
    scikit-learn             1.3.2
    ```

- tools requirements

    ```bash
    vamc                     >= 2.x
    vsx                      2.3.6b7
    tvm                      2.0.0
    ```