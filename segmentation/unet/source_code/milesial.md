## milesial

### step.1 获取预训练模型

```
link：https://github.com/milesial/Pytorch-UNet
branch: master
commit: a013c80ca6b011ba34ba0700b961b77e0ed003a2
```

根据原始仓库即可进模型导出：
- 在[predict.py#L24](https://github.com/milesial/Pytorch-UNet/blob/master/predict.py#L24)，增加torchscript和onnx转换代码：
    ```python
    dump_input = torch.rand((1, 3, 512, 512))
    export_onnx_file = args.model.replace(".pth",".onnx")
    torch.onnx.export(net.cpu(), dump_input.cpu(), export_onnx_file, verbose=True, opset_version=11, input_names=["input"])

    traced_script = torch.jit.trace(net, dump_input)
    traced_script.save(args.model.replace(".pth",".torchscript.pt"))
    ```


### step.2 准备数据集
- 下载[carvana](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data)数据集，解压


### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[milesial-config.yaml](../vacc_code/build/milesial-config.yaml)
3. 执行转换

   ```bash
    vamc build ../vacc_code/build/milesial-config.yaml
   ```

### step.4 benchmark
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path carvana/imgs \
    --target_path  carvana/imgs_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[milesial-unet_scale0.5-vdsp_params.json](../vacc_code/vdsp_params/milesial-unet_scale0.5-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/unet_scale0.5-int8-kl_divergence-3_512_512-vacc/unet_scale0.5 \
    --vdsp_params ../vacc_code/vdsp_params/milesial-unet_scale0.5-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,512,512]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/unet_scale0.5-int8-kl_divergence-3_512_512-vacc/unet_scale0.5 \
    --vdsp_params vacc_code/vdsp_params/milesial-unet_scale0.5-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,512,512] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [milesial-vamp_eval.py](../vacc_code/vdsp_params/milesial-vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/milesial-vamp_eval.py \
    --src_dir carvana/imgs \
    --gt_dir carvana/masks \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 512 512 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```