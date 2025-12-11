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
- 下载[GoPro](https://seungjunnah.github.io/Datasets/gopro)数据集
- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [basicsr_edsr.yaml](../build_in/build/basicsr_edsr.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd edsr
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/basicsr_edsr.yaml
    ```

### step.4 模型推理

    - 参考[vsx脚本](../build_in/vsx/python/basicsr_vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/basicsr_vsx_inference.py \
        --lr_image_dir  /path/to/DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --model_prefix_path deploy_weights/basicsr_edsr_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/basicsr-edsr_mx2-vdsp_params.json \
        --hr_image_dir /path/to/DIV2K/DIV2K_valid_HR \
        --save_dir ./infer_output \
        --device 0
    ```
    
    - 测试精度在打印信息中，如下：
    ```
    # fp16
    mean psnr: 33.49612039932567, mean ssim: 0.9292504484925148

    # int8 
    mean psnr: 32.752828952396214, mean ssim: 0.9175428974586191
    ```
    
### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[basicsr-edsr_m2x-vdsp_params.json](../build_in/vdsp_params/basicsr-edsr_mx2-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/basicsr_edsr_int8/mod \
    --vdsp_params ../build_in/vdsp_params/basicsr-edsr_mx2-vdsp_params.json \
    -i 1 p 1 -b 1
    ```
    
2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；
   
    - 数据准备，参考[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path DIV2K/DIV2K_valid_LR_bicubic/X2 \
    --target_path  DIV2K/DIV2K_valid_LR_bicubic/X2_npz \
    --text_path npz_datalist.txt
    ```

    - vamp推理，获得npz结果
    ```bash
    vamp -m deploy_weights/basicsr_edsr_int8/mod \
    --vdsp_params ../build_in/vdsp_params/basicsr-edsr_mx2-vdsp_params.json \
    -i 1 p 1 -b 1 \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```

    - 解析npz结果，统计精度：[basicsr-vamp_eval.py](../build_in/vdsp_params/basicsr-vamp_eval.py)
    ```bash
    python ../build_in/vdsp_params/basicsr-vamp_eval.py \
    --gt_dir DIV2K/DIV2K_valid_HR \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir npz_output \
    --input_shape 768 1024 \
    --draw_dir npz_draw_result \
    --vamp_flag
    ```

## Tips
- EDSR_x1，[edsr_arch_modify.py](./edsr_arch_modify.py)，在GoPRo去模糊数据集上训练，尺寸未放大，内部定义为：Deblur
