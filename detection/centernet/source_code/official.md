
### official

#### step.1 获取预训练模型

```bash
git clone https://github.com/xingyizhou/CenterNet.git
cd CenterNet
git checkout 4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c
```

官方未提供onnx转换脚本，可以在项目'src/lib/detectors/ctdet.py'文件30行插入如下代码
```python
input_names = ["input"]
output_names = ["output"]
inputs = torch.randn(1, 3, self.opt.input_h, self.opt.input_w)

torch_out = torch.onnx._export(self.model, inputs, 'centernet.onnx', export_params=True, verbose=False,
                            input_names=input_names, output_names=output_names, opset_version=10)
```

然后，运行项目demo即可

```bash
cd src

python demo.py ctdet --demo ../images/16004479832_a748d55f21_k.jpg --load_model ../models/model_best.pth --arch res_18 --gpus -1 --fix_res
```

### step.2 准备数据集
- 准备[COCO](https://cocodataset.org/#download)数据集

### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[official_config.yaml](../vacc_code/build/official_config.yaml)：
    ```bash
    vamc build ../vacc_code/build/official_config.yaml
    ```

### step.4 benchmark
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path path/to/coco_val2017 \
    --target_path  path/to/coco_val2017_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，修改vdsp参数[official-centernet_res18-vdsp_params.json](../vacc_code/vdsp_params/official-centernet_res18-vdsp_params.json)：
    ```bash
    vamp -m deploy_weights/centernet_res18-int8-kl_divergence-3_512_512-vacc/centernet_res18 \
    --vdsp_params ../vacc_code/vdsp_params/official-centernet_res18-vdsp_params.json \
    -i 2 p 2 -b 1 -s [1,512,512]
    ```
4. 精度测试：
    ```bash
    vamp -m deploy_weights/centernet_res18-int8-kl_divergence-3_512_512-vacc/centernet_res18 \
    --vdsp_params ../vacc_code/vdsp_params/official-centernet_res18-vdsp_params.json \
    -i 2 p 2 -b 1 -s [1,512,512] \
    --datalist datalist_npz.txt \
    --path_output outputs/centernet
    ```
5. 基于[vamp_decode.py](../vacc_code/vdsp_params/vamp_decode.py)，解析vamp输出的npz文件，进行绘图和保存txt结果：
    ```bash
    python ../vacc_code/vdsp_params/vamp_decode.py \
    --gt_dir datasets/coco_val2017 \
    --input_npz_path datalist_npz.txt \
    --out_npz_dir outputs/centernet \
    --draw_dir coco_val2017_npz_result
    ```
6. 基于[eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标：
   ```bash
    python ../../common/eval/eval_map.py \
    --gt path/to/instances_val2017.json \
    --txt path/to/vamp_draw_output
   ```

### Tips
- 此模型为forward形式，后处理需调用原始仓库相关代码，注意编译`soft_nms`