
### mmdet

#### step.1 获取预训练模型

```bash
git clone https://github.com/open-mmlab/mmdetection
cd mmdetection
git checkout ca11860f4f3c3ca2ce8340e2686eeaec05b29111
```

mmdet提供了将原始torch模型转为onnx格式的脚本，运行以下命令即可

```bash
cd mmdet

python tools/deployment/pytorch2onnx.py configs/centernet/centernet_resnet18_140e_coco.py models/centernet/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth  --output-file models/centernet/centernet-512.onnx --shape 512 512 --verify --skip-postprocess
```

### step.2 准备数据集
- 准备[COCO](https://cocodataset.org/#download)数据集

### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[mmdet_config.yaml](../vacc_code/build/mmdet_config.yaml)：
    ```bash
    vamc build ../vacc_code/build/mmdet_config.yaml
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
3. 性能测试，修改vdsp参数[mmdet-centernet_res18-vdsp_params.json](../vacc_code/vdsp_params/mmdet-centernet_res18-vdsp_params.json)：
    ```bash
    vamp -m deploy_weights/centernet_res18-int8-kl_divergence-3_512_512-vacc/centernet_res18 \
    --vdsp_params ../vacc_code/vdsp_params/mmdet-centernet_res18-vdsp_params.json \
    -i 2 p 2 -b 1 -s [1,512,512]
    ```
4. 精度测试：
    ```bash
    vamp -m deploy_weights/centernet_res18-int8-kl_divergence-3_512_512-vacc/centernet_res18 \
    --vdsp_params ../vacc_code/vdsp_params/mmdet-centernet_res18-vdsp_params.json \
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
- 此模型为forward形式，后处理需调用了official实现的相关代码，注意编译`soft_nms`
- 注意和official实现不一样的地方是预处理的均值方差，配置vdsp_params.json时需注意