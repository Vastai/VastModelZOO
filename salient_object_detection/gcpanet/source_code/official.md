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
- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试图像转换为对应npz文件

### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[config.yaml](../vacc_code/build/config.yaml)
3. 执行转换

   ```bash
   vamc build ../vacc_code/build/config.yaml
   ```

### step.4 benchmark
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path sod/ECSSD/image \
    --target_path sod/ECSSD/image_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[official-gcpanet-vdsp_params.json](../vacc_code/vdsp_params/official-gcpanet-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/gcpanet-int8-kl_divergence-3_512_512-vacc/gcpanet \
    --vdsp_params vacc_code/vdsp_params/official-gcpanet-vdsp_params.json \
    -i 1 p 1 -b 1
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/gcpanet-int8-kl_divergence-3_512_512-vacc/gcpanet \
    --vdsp_params vacc_code/vdsp_params/official-gcpanet-vdsp_params.json \
    -i 1 p 1 -b 1 \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --src_dir data/ECSSD/image \
    --gt_dir data/ECSSD/mask \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir npz_output \
    --input_shape 512 512 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```
6. 基于[eval.py](../../common/eval/eval.py)，统计精度信息
   - 来自[PySODEvalToolkit](https://github.com/lartpang/PySODEvalToolkit)工具箱
   - 配置数据集路径：[config_dataset.json](../../common/eval/examples/config_dataset.json)
   - 配置模型推理结果路径及图片格式：[config_method.json](../../common/eval/examples/config_method.json)
   - 执行评估：`python eval.py --dataset-json path/to/config_dataset.json --method-json path/to/source_code/config_method.json`，即可获得多个精度评估信息

### Tips
