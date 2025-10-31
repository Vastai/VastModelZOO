
### step.1 获取预训练模型

```bash
gitlab: https://github.com/Star-Clouds/CenterFace
branch: master
commit: b82ec0c
```

- 从源仓库下载onnx：https://github.com/Star-Clouds/CenterFace/tree/master/models/onnx


### step.2 准备数据集
- [校准数据集](https://huggingface.co/datasets/wider_face/blob/main/data/WIDER_val.zip)
- [评估数据集](https://huggingface.co/datasets/wider_face/blob/main/data/WIDER_val.zip)


### step.3 模型转换
1. 根据具体模型修改配置文件
    - [pytorch_centerface.yaml](../build_in/build/pytorch_centerface.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 执行转换
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/pytorch_centerface.yaml
    ```

### step.4 模型推理

1. 参考[vsx脚本](../build_in/vsx/python/vsx.py)，修改参数并运行如下脚本
    ```bash
    python ../build_in/vsx/python/vsx.py \
        --file_path  /path/to/widerface/val/ \
        --model_prefix_path deploy_weights/pytorch_centerface_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-retinaface_resnet50-vdsp_params.json \
        --save_dir ./runstream_output \
        --device 0
    ```
    - 注意替换命令行中--file_path为实际路径


### step.5 性能测试
1. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`widerface_npz_list.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path /path/to/widerface/val/ --target_path /path/to/widerface_npz --text_path widerface_npz_list.txt
    ```

2. 性能测试
    ```bash
    vamp -m deploy_weights/pytorch_centerface_run_stream_fp16/mod --vdsp_params ../build_in/vdsp_params/official-retinaface_resnet50-vdsp_params.json -i 2 p 2 -b 1
    ```

## Tips
- int8可编译，推理报错，待查：`Exception: run err, err code: 803012, vafwER_MCU_ERROR_LMCU0`