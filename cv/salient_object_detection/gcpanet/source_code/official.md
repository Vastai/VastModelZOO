## Official

```
link: https://github.com/JosephChenHub/GCPANet
branch: master
commit: 8ab0d6dcfe3e9f14f7853393425745136c98a950
```

### step.1 获取预训练模型
- 观察原仓库[test.py#L59](https://github.com/JosephChenHub/GCPANet/blob/master/test.py#L59)，模型forward后只使用到第一个返回值。为减少模型推理时的数据拷贝，修改[net.py#L240](https://github.com/JosephChenHub/GCPANet/blob/master/net.py#L240)，只返回第一值，`return out2`
- 在原仓库[test.py#L49](https://github.com/JosephChenHub/GCPANet/blob/master/test.py#L49)，定义模型和加载训练权重后，添加以下脚本，执行即可导出onnx和torchscript：
```python
input_shape = (1, 3, 512, 512)
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)
with torch.no_grad():
    scripted_model = torch.jit.trace(self.net, input_data).eval()
    scripted_model.save(self.snapshot.replace(".pt", ".torchscript.pt"))
    scripted_model = torch.jit.load(self.snapshot.replace(".pt", ".torchscript.pt"))

    torch.onnx.export(self.net, input_data, self.snapshot.replace(".pt", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
    dynamic_axes= {
        "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
        "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}
        }
    )
```


### step.2 准备数据集
- 下载[ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)数据集


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_gcpanet.yaml](../build_in/build/official_gcpanet.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd gcpanet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_gcpanet.yaml
    ```

### step.4 模型推理

- 参考：[official_vsx_inference.py](../build_in/vsx/python/official_vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/official_vsx_inference.py \
        --image_dir  /path/to/sod/ECSSD/image  \
        --model_prefix_path deploy_weights/official_gcpanet_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-gcpanet-vdsp_params.json \
        --save_dir ./infer_output \
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

    - fp16精度：
    Dataset: ECSSD

    | methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
    |-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
    | Method1   | 0.113 |         0.775 |         0.765 |         0.771 |          0.847 |          0.798 |           1 |        0.78 |   0.842 |   0.833 |   0.843 | 0.788 | 0.731 |

    - int8精度：
    Dataset: ECSSD

    | methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
    |-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
    | Method1   | 0.052 |         0.908 |         0.892 |         0.895 |          0.983 |          0.925 |           1 |       0.871 |   0.929 |   0.916 |   0.923 | 0.896 | 0.866 |

    </details>

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[official-gcpanet-vdsp_params.json](../build_in/vdsp_params/official-gcpanet-vdsp_params.json)
    ```
    vamp -m deploy_weights/official_gcpanet_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-gcpanet-vdsp_params.json \
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
    vamp -m deploy_weights/official_gcpanet_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-gcpanet-vdsp_params.json \
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
        --input_shape 512 512 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```

    - 统计精度信息，基于[eval.py](../../common/eval/eval.py)
        ```
        python ../../common/eval/eval.py --dataset-json path/to/config_dataset.json --method-json path/to/config_method.json
        ```
        - 来自[PySODEvalToolkit](https://github.com/lartpang/PySODEvalToolkit)工具箱
        - 配置数据集路径：[config_dataset.json](../../common/eval/examples/config_dataset.json)
        - 配置模型推理结果路径及图片格式：[config_method.json](../../common/eval/examples/config_method.json)
    

