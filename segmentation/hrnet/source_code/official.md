## official

### step.1 获取预训练模型

```
link: https://github.com/HRNet/HRNet-Semantic-Segmentation
branch: pytorch-v1.1
commit: 88419ab18813f2c9193985e2d4d31d3d07abe839
```

- 克隆官方仓库后，将[export.py](./official/export.py)文件移动至 `{git_dir}/tools`目录下，配置相应参数，执行即可获得onnx和torchscript

### step.2 准备数据集

- 下载[cityscapes](https://www.cityscapes-dataset.com/)数据集，解压
- 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的 `npz_datalist.txt`

### step.3 模型转换

1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[config.yaml](../vacc_code/build/config.yaml)
3. 执行转换

   ```bash
    vamc build ../vacc_code/build/config.yaml
   ```

### step.4 benchmark

1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的 `npz_datalist.txt`
   ```bash
   python ../../common/utils/image2npz.py \
   --dataset_path cityscapes/leftImg8bit/val \
   --target_path  cityscapes/leftImg8bit/val_npz \
   --text_path npz_datalist.txt
   ```
3. 性能测试，配置vdsp参数[official-hrnet_w48-vdsp_params.json](../vacc_code/vdsp_params/official-hrnet_w48-vdsp_params.json)，执行：
   ```bash
   vamp -m deploy_weights/hrnet_w48-int8-kl_divergence-3_512_512-vacc/hrnet_w48 \
   --vdsp_params vacc_code/vdsp_params/official-hrnet_w48-vdsp_params.json \
   -i 1 p 1 -b 1
   ```
4. 精度测试，推理得到npz结果：
   ```bash
   vamp -m deploy_weights/hrnet_w48-int8-kl_divergence-3_512_512-vacc/hrnet_w48 \
   --vdsp_params vacc_code/vdsp_params/official-hrnet_w48-vdsp_params.json \
   -i 1 p 1 -b 1 \
   --datalist npz_datalist.txt \
   --path_output npz_output
   ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --src_dir cityscapes/leftImg8bit/val \
    --gt_dir cityscapes/gtFine/val \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 512 512 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```

### Tips

- [HRNet-OCR](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR)暂不支持