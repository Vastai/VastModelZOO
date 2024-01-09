## bubbliiiing

### step.1 获取预训练模型

```
link：https://github.com/bubbliiiing/unet-pytorch
branch: main
commit: 8ab373232c5c3a1877f3e84e6f5d97404089c20f
```

根据原始仓库即可进模型导出：
- 将[predict.py#L26](https://github.com/bubbliiiing/unet-pytorch/blob/main/predict.py#L26)修改为`export_onnx`模式
- 在[unet.py#L267](https://github.com/bubbliiiing/unet-pytorch/blob/main/unet.py#L267)，增加torchscript转换代码：
    ```python
    scripted_model = torch.jit.trace(self.net, im).eval()
    torch.jit.save(scripted_model, model_path.replace(".onnx", ".torchscript.pt"))
    ```
- 执行[predict.py](https://github.com/bubbliiiing/unet-pytorch/blob/main/predict.py#L26)即可导出onnx和torchscript（opset_version=11）

> Tips
>
> opset_version=10时转onnx报错：RuntimeError: ONNX export failed: Couldn't export operator aten::upsample_bilinear2d


### step.2 准备数据集
- 下载[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)数据集，解压
- 使用[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，提取val图像数据集和转换为npz格式

### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[bubbliiiing-config.yaml](../vacc_code/build/bubbliiiing-config.yaml)
3. 执行转换

   ```bash
    vamc build ../vacc_code/build/bubbliiiing-config.yaml
   ```

### step.4 benchmark
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`，注意只转换`VOC2012/ImageSets/Segmentation/val.txt`对应的验证集图像（配置相应路径）：
    ```bash
    python ../vacc_code/vdsp_params/image2npz.py \
    --dataset_path VOC2012/JPEGImages \
    --target_path  VOC2012/JPEGImages_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[bubbliiiing-unet_resnet50-vdsp_params.json](../vacc_code/vdsp_params/bubbliiiing-unet_resnet50-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/unet_resnet50-int8-kl_divergence-3_256_256-vacc/unet_resnet50 \
    --vdsp_params ../vacc_code/vdsp_params/bubbliiiing-unet_resnet50-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,256,256]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/unet_resnet50-int8-kl_divergence-3_256_256-vacc/unet_resnet50 \
    --vdsp_params vacc_code/vdsp_params/bubbliiiing-unet_resnet50-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,256,256] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [bubbliiiing-vamp_eval.py](../vacc_code/vdsp_params/bubbliiiing-vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/bubbliiiing-vamp_eval.py \
    --src_dir VOC2012/JPEGImages_val \
    --gt_dir VOC2012/SegmentationClass \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 256 256 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


### Tips
