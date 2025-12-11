## Official来源

```
github: https://github.com/xuebinqin/U-2-Net/tree/master
branch: master
commit: 53dc9da026650663fc8d8043f3681de76e91cfde
```

### step.1 获取预训练模型
- `u2net`原始模型forward内返回7个特征图，[model/u2net.py#L420](https://github.com/xuebinqin/U-2-Net/blob/master/model/u2net.py#L420)；此处只需第一个，手动修改L420为`return F.sigmoid(d0)`
- 将onnx转换脚本[export_onnx.py](./export_onnx.py)置于仓库根目录，执行此脚本导出onnx模型


### step.2 准备数据集
- 下载[ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)数据集

- 下载[Supervisely Person](https://ecosystem.supervise.ly/projects/persons/)数据集，解压
- 按[link](https://blog.csdn.net/u011622208/article/details/108535943)整理转换数据集


### step.3 模型转换

1. 根据具体模型,修改配置文件
    - [official_u2net.yaml](../build_in/build/official_u2net.yaml)

2. 执行转换
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    cd u2net
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_u2net.yaml
    ```

### step.4 模型推理

1. 参考[vsx脚本](../build_in/vsx/python/official_vsx_inference.py)，修改参数并运行如下脚本
    ```bash
    python ../build_in/vsx/python/official_vsx_inference.py \
        --image_dir  /path/to/ECSSD/image \
        --model_prefix_path deploy_weights/official_u2net_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-u2net-vdsp_params.json \
        --save_dir ./infer_output \
        --device 0
    ```
    - 注意替换命令行中--file_path为实际路径

2. 基于[eval.py](../../common/eval/eval.py)，统计精度信息
   - 来自[PySODEvalToolkit](https://github.com/lartpang/PySODEvalToolkit)工具箱
   - 拷贝[config_dataset.json](../../common/eval/examples/config_dataset.json)到当前路径，修改配置文件中数据集路径
   - 拷贝[config_method.json](../../common/eval/examples/config_method.json)到当前路径，修改path的值/path/to/infer_output
   - 执行评估命令：
   ```
   python ../../common/eval/eval.py --dataset-json ./config_dataset.json --method-json ./config_method.json
   ```
    - 测试精度信息，如下：
    Dataset: ECSSD

    | methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
    |-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
    | Method1   |  0.04 |         0.928 |         0.906 |         0.909 |          0.975 |          0.922 |           1 |       0.906 |   0.949 |   0.937 |   0.945 | 0.918 |  0.89 |

### step.5 性能精度测试
1. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path /path/to/ECSSD/image \
    --target_path /path/to/ECSSD/image_npz \
    --text_path npz_datalist.txt
    ```

2. 性能测试，配置vdsp参数[official-u2net-vdsp_params.json](../build_in/vdsp_params/official-u2net-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/official_u2net_int8/mod \
    --vdsp_params ../build_in/vdsp_params/official-u2net-vdsp_params.json \
    -i 1 p 1 -b 1
    ```

3. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/official_u2net_int8/mod \
    --vdsp_params ../build_in/vdsp_params/official-u2net-vdsp_params.json \
    -i 1 p 1 -b 1 \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
    
4. [official_vamp_eval.py](../build_in/vdsp_params/official_vamp_eval.py)，解析npz结果，绘图：
   ```bash
    python ../build_in/vdsp_params/official_vamp_eval.py \
    --src_dir /path/to/ECSSD/image \
    --gt_dir /path/to/ECSSD/mask \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir npz_output \
    --input_shape 320 320 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```

