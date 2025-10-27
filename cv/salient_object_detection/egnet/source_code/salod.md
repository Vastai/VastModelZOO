## SALOD来源

```
# SALOD
link: https://github.com/moothes/SALOD
branch: master
commit: 59a4b280463ac9519420ad87c4f6666414f20aed
```

### step.1 获取预训练模型
- 观察原仓库[test.py#L34](https://github.com/moothes/SALOD/blob/master/test.py#L34)，模型forward后只使用到一个字典的`['final']`值。为减少模型推理时的数据拷贝，修改模型返回值[methods/egnet.py#L184](https://github.com/moothes/SALOD/blob/master/methods/egnet.py#L184)，只返回一个值，`return up_sal_final[-1]`
- 在原仓库[test.py#L94](https://github.com/moothes/SALOD/blob/master/test.py#L94)，定义模型和加载训练权重后，添加以下脚本，执行即可导出onnx和torchscript：
```python
input_shape = (1, 3, 320, 320)
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)
with torch.no_grad():
    scripted_model = torch.jit.trace(model, input_data).eval()
    scripted_model.save(config['weight'].replace(".pth", ".torchscript.pt"))
    scripted_model = torch.jit.load(config['weight'].replace(".pth", ".torchscript.pt"))

    torch.onnx.export(model, input_data, config['weight'].replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
    dynamic_axes= {
        "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
        "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'},
        }
    )
```


### step.2 准备数据集
- 下载[ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)数据集

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_egnet.yaml](../build_in/build/official_egnet.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd dhsnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_egnet.yaml
    ```

### step.4 模型推理
1. runstream
    - 参考：[salod_vsx_inference.py](../build_in/vsx/python/salod_vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/salod_vsx_inference.py \
        --image_dir /path/to/sod/ECSSD/image \
        --model_prefix_path deploy_weights/official_egnet_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/salod-egnet-vdsp_params.json \
        --save_dir ./runstream_output \
        --device 0
    ```

    - 统计精度信息，基于[eval.py](../../common/eval/eval.py)
        ```
        python ../../common/eval/eval.py --dataset-json path/to/config_dataset.json --method-json path/to/config_method.json
        ```
        - 来自[PySODEvalToolkit](https://github.com/lartpang/PySODEvalToolkit)工具箱
        - 配置数据集路径：[config_dataset.json](../../common/eval/examples/config_dataset.json)
        - 配置模型推理结果路径及图片格式：[config_method.json](../../common/eval/examples/config_method.json)
    
    <details><summary>点击查看精度统计结果</summary>

    - fp16精度
    Dataset: ECSSD

    | methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
    |-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
    | Method1   |  0.06 |          0.89 |         0.876 |         0.872 |           0.95 |          0.892 |           1 |       0.882 |   0.921 |   0.912 |   0.909 | 0.884 | 0.854 |

    - int8精度
    Dataset: ECSSD

    | methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
    |-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
    | Method1   | 0.047 |         0.917 |         0.902 |         0.898 |          0.977 |          0.915 |           1 |       0.906 |    0.94 |    0.93 |   0.925 | 0.907 | 0.882 |

    </details>

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[salod-egnet-vdsp_params.json](../build_in/vdsp_params/salod-egnet-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_egnet_run_stream_int8/mod \
        --vdsp_params ../build_in/vdsp_params/salod-egnet-vdsp_params.json \
        -i 1 p 1 -b 1
    ```

2. 精度测试
    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path /path/to/ECSSD/image \
        --target_path /path/to/ECSSD/image_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/official_egnet_run_stream_int8/mod \
        --vdsp_params ../build_in/vdsp_params/salod-egnet-vdsp_params.json \
        -i 1 p 1 -b 1 \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果，参考：[vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)，
    ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
        --src_dir /path/to/ECSSD/image \
        --gt_dir /path/to/ECSSD/mask \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output \
        --input_shape 320 320 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```
    
    - 统计精度信息，基于[eval.py](../../common/eval/eval.py)
        ```
        python eval.py --dataset-json path/to/config_dataset.json --method-json path/to/config_method.json
        ```
        - 来自[PySODEvalToolkit](https://github.com/lartpang/PySODEvalToolkit)工具箱
        - 配置数据集路径：[config_dataset.json](../../common/eval/examples/config_dataset.json)
        - 配置模型推理结果路径及图片格式：[config_method.json](../../common/eval/examples/config_method.json)
        
