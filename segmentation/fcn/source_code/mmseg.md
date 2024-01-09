## mmsegmentation

### step.1 获取预训练模型

```
link: https://github.com/open-mmlab/mmsegmentation
branch: v1.0.0rc2
commit: 8a611e122d67b1d36c7929331b6ff53a8c98f539
```

使用mmseg转换代码[pytorch2torchscript.py](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/pytorch2torchscript.py)，命令如下

```bash
python tools/pytorch2torchscript.py  \
    configs/fcn/fcn_r50-d8_512x512_20k_voc12aug.py \
    --checkpoint ./pretrained/mmseg/ann/fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth \
    --output-file ./onnx/mmseg/ann/torchscript/fcn_r50_d8_20k-512.torchscript.pt \
    --shape 512 512
```

### step.2 准备数据集

- 下载[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)数据集，解压
- 使用[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，提取val图像数据集和转换为npz格式

### step.3 模型转换

1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[mmseg_config.yaml](../vacc_code/build/mmseg_config.yaml)
3. 执行转换

   ```bash
    vamc build ../vacc_code/build/mmseg_config.yaml
   ```

### step.4 benchmark

1. 获取vamp性能测试工具
2. 基于[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的 `npz_datalist.txt`，注意只转换 `VOC2012/ImageSets/Segmentation/val.txt`对应的验证集图像（配置相应路径）：
   ```bash
   python ../vacc_code/vdsp_params/image2npz.py \
   --dataset_path VOC2012/JPEGImages \
   --target_path  VOC2012/JPEGImages_npz \
   --text_path npz_datalist.txt
   ```
3. 性能测试，配置vdsp参数[mmseg-fcn_r50_d8_20k-vdsp_params.json](../vacc_code/vdsp_params/mmseg-fcn_r50_d8_20k-vdsp_params.json)，执行：
   ```bash
   vamp -m deploy_weights/fcn_r50_d8_20k-int8-kl_divergence-3_512_512-vacc/fcn_r50_d8_20k \
   --vdsp_params ../vacc_code/vdsp_params/mmseg-fcn_r50_d8_20k-vdsp_params.json \
   -i 2 p 2 -b 1 -s [3,512,512]
   ```
4. 精度测试，推理得到npz结果：
   ```bash
   vamp -m deploy_weights/fcn_r50_d8_20k-int8-kl_divergence-3_512_512-vacc/fcn_r50_d8_20k \
   --vdsp_params vacc_code/vdsp_params/mmseg-fcn_r50_d8_20k-vdsp_params.json \
   -i 2 p 2 -b 1 -s [3,512,512] \
   --datalist npz_datalist.txt \
   --path_output npz_output
   ```
5. [mmseg_vamp_eval.py](../vacc_code/vdsp_params/mmseg_vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/mmseg_vamp_eval.py \
    --src_dir VOC2012/JPEGImages_val \
    --gt_dir VOC2012/SegmentationClass \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 512 512 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```
