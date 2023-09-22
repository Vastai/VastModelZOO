## BasicSR

### step.1 获取预训练模型

```
link: https://github.com/XPixelGroup/BasicSR
branch: v1.4.2
commit: 651835a1b9d38dbbdaf45750f56906be2364f01a
```
- 拉取原始仓库，EDSR_Mx2模型使用配置表[options/test/EDSR/test_EDSR_Mx2.yml](https://github.com/XPixelGroup/BasicSR/blob/master/options/test/EDSR/test_EDSR_Mx2.yml)，在原仓库[ModelZoo_CN.md](https://github.com/XPixelGroup/BasicSR/blob/v1.4.2/docs/ModelZoo_CN.md#图像超分官方模型)下载对应预训练权重；
- 在原仓库[basicsr/models/sr_model.py#L31](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/models/sr_model.py#L31)，增加转换代码：
    ```python
    model = self.net_g
    checkpoint = "EDSR_Mx2_f64b16_DIV2K_official-3ba7b086.pth"
    input_shape = (1, 3, 768, 1024)
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, input_data)
        scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))
        scripted_model = torch.jit.load(checkpoint.replace(".pth", ".torchscript.pt"))

    # onnx==10.0.0，opset 13
    import onnx
    with torch.no_grad():
       torch.onnx.export(model, input_data, checkpoint.replace(".pth", "_dynamic.onnx"), input_names=["input"], output_names=["output"], opset_version=13,
       dynamic_axes= {
                   "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
                   "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
       )
    ```

- 上述修改后，修改[basicsr/test.py](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/test.py)配置表路径，启动此脚本即可完成前向推理和模型导出

- EDSR_1x，可参考[export.py](./basicsr/export.py)转换，对应参数表[test_EDSR_x1.yml](./basicsr/test_EDSR_x1.yml)


### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集

- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件




### step.3 模型转换
1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[basicsr-config.yaml](../vacc_code/build/basicsr-config.yaml)，执行转换命令：
    ```bash
    vamc build ./vacc_code/build/basicsr-config.yaml
    ```
### step.4 benchmark

1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path DIV2K/DIV2K_valid_LR_bicubic/X2 \
    --target_path  DIV2K/DIV2K_valid_LR_bicubic/X2_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[basicsr-edsr_m2x-vdsp_params.json](../vacc_code/vdsp_params/basicsr-edsr_mx2-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/EDSR_Mx2-int8-percentile-3_768_1024-vacc/EDSR_Mx2 \
    --vdsp_params ../vacc_code/vdsp_params/basicsr-edsr_mx2-vdsp_params.json \
    -i 1 p 1 -b 1
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/EDSR_Mx2-int8-percentile-3_768_1024-vacc/EDSR_Mx2 \
    --vdsp_params ../vacc_code/vdsp_params/basicsr-edsr_mx2-vdsp_params.json \
    -i 1 p 1 -b 1 \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [basicsr-vamp_eval.py](../vacc_code/vdsp_params/basicsr-vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/basicsr-vamp_eval.py \
    --gt_dir DIV2K/DIV2K_valid_HR \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/rcan \
    --input_shape 768 1024 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


## Tips
- 关于预处理，发现模型forward内已存在减均值操作[edsr_arch_modify.py#L499](./basicsr/edsr_arch_modify.py#L499)，所以build时只需要除255；以此推算runstream的vdsp预处理内`norma_type`应该设置为`DIV255 (6)`，但实际测试发现精度极差，设置为`MINUSMEAN_DIVSTD (3)`却正常，暂无法定位原因
- 原始仓库的模型均比较大，在较大尺寸quant时可能会报错：已中断（系统内存问题，需切换至内存较大机器上）
- EDSR_x1，[edsr_arch_modify.py](./edsr_arch_modify.py)，在GoPRo去模糊数据集上训练，尺寸未放大，内部定义为：Deblur
- 无法导出onnx，含pixel_shuffle(opset_version=13 is ok)，只能导出torchscript
- EDSR_Mx3，int8 build通过，但run时报错：
    ```bash
    [20230331 13:35:41.958][ERROR]pwrite(5, 0x55b2cb352680, 0x400, 0x87fffc000) error: Input/output error.
    TVMError: vacc error: vaccrt_copy_weight(device_id, name, device_ptr->key, host_ptr, total_size, direction) failed with error: Failed to copy data from host to device(DDR)
    ```
