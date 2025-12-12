## Official

```
link: https://github.com/chaofengc/Face-SPARNet
branch: master
commit: 5204283bcb584067a4b28e44231cf8f150cf07a3
```

### step.1 获取预训练模型
一般在原始仓库内进行模型转为onnx或torchscript。在原仓库test或val脚本内，如[test.py#L36](https://github.com/chaofengc/Face-SPARNet/blob/master/test.py#L36)，定义模型和加载训练权重后，添加以下脚本可实现：
```python

input_shape = (1, 3, 128, 128)
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(network, input_data).eval()
torch.jit.save(scripted_model, 'SPARNet-Light-Attn3D.torchscript.pt')

import onnx
torch.onnx.export(network, input_data, 'SPARNet-Light-Attn3D.onnx', input_names=["input"], output_names=["output"], opset_version=10,
            # dynamic_axes= {
            #                 "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
            #                 "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}
            #                 }
)
```

SPARNetHD-Attn3D模型使用[test_enhance_single_unalign.py#L53](https://github.com/chaofengc/Face-SPARNet/blob/master/test_enhance_single_unalign.py#L53)脚本转换，同时该子模型使用了生成对抗网络，其鉴别器模块用到频谱归一化函数spectral_norm，其中包含了torch.mv、 torch.dot算子，转onnx和torchscript会出现错误。参考[pytorch](https://github.com/pytorch/pytorch/issues/27723)，按官方建议，使用remove_all_spectral_norm，移除spectral_norm的函数，可正常转为onnx和torchscript
```python
import torch.nn as nn
def remove_all_spectral_norm(item):
    if isinstance(item, nn.Module):
        try:
            nn.utils.remove_spectral_norm(item)
        except Exception:
            pass
        
        for child in item.children():  
            remove_all_spectral_norm(child)

    if isinstance(item, nn.ModuleList):
        for module in item:
            remove_all_spectral_norm(module)

    if isinstance(item, nn.Sequential):
        modules = item.children()
        for module in modules:
            remove_all_spectral_norm(module)


network = model.netG
network.eval()

remove_all_spectral_norm(network)

input_shape = (1, 3, 512, 512)
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(network, input_data).eval()
torch.jit.save(scripted_model, 'SPARNetHD-Attn3D.torchscript.pt')

import onnx
torch.onnx.export(network, input_data, 'SPARNetHD-Attn3D.onnx', input_names=["input"], output_names=["output"], opset_version=10,
            # dynamic_axes= {
            #                 "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
            #                 "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}
            #                 }
)
```


### step.2 准备数据集
- 参考源仓库，[Face-SPARNet](https://github.com/chaofengc/Face-SPARNet#sparnet)，在数据准备步骤，下载`test_dirs.tgz`，解压获取`Helen_test_DIC`数据集

> **Note**
> 
> 测试高清HR图像(3x128x128)，使用`bicubic`下采样至16x16，构建为测试低清LR图像 (3x16x16)
> 
> 测试低清LR图像npz：通过[image2npz.py](../build_in/vdsp_params/image2npz.py)生成，已转换至LR图像

### step.3 模型转换
1. 根据具体模型修改配置文件
    - [config.yaml](../build_in/build/config.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd sparnet
    mkdir workspace
    cd workspace
   vamc build ../build_in/build/config.yaml
   ```

### step.4 模型推理

- 参考[vsx脚本](../build_in/vsx/official_vsx_inference.py)
    ```bash
    python ../build_in/vsx/official_vsx_inference.py \
        --lr_image_dir  /path/to/code/model_check/SR/face/Face-SPARNet/datasets/test_dirs/Helen_test_DIC/LR \
        --model_prefix_path deploy_weights/sparnet_light_attn3d/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-sparnet_light_attn3d-vdsp_params.json \
        --hr_image_dir /path/to//code/model_check/SR/face/Face-SPARNet/datasets/test_dirs/Helen_test_DIC/HR \
        --save_dir ./infer_output \
        --device 0
    ```

### step.5 性能精度测试

1. 基于[image2npz.py](../build_in/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`（注意，HR图像采用`bicubic`下采样至16x16作为LR图像）：
    ```bash
    python ../build_in/vdsp_params/image2npz.py \
    --hr_path /path/to/Helen_test_DIC/HR \
    --lr_path /path/to/Helen_test_DIC/LR \
    --target_path /path/to/Helen_test_DIC/LR_npz \
    --text_path npz_datalist.txt
    ```
2. 性能测试，配置vdsp参数[official-sparnet_light_attn3d-vdsp_params.json](../build_in/vdsp_params/official-sparnet_light_attn3d-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/sparnet_light_attn3d-int8-kl_divergence-3_128_128-vacc/sparnet_light_attn3d \
    --vdsp_params ../build_in/vdsp_params/official-sparnet_light_attn3d-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128]
    ```
3. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/sparnet_light_attn3d-int8-kl_divergence-3_128_128-vacc/sparnet_light_attn3d \
    --vdsp_params ../build_in/vdsp_params/official-sparnet_light_attn3d-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
4. [vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
    --gt_dir /path/to/Helen_test_DIC/HR \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir npz_output \
    --input_shape 128 128 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


#### Tips
- 验证数据集Helen_test_DIC，参考源仓库[Face-SPARNet](https://github.com/chaofengc/Face-SPARNet#download-pretrain-models-and-dataset)数据准备，先将HR图像使用`bicubic`下采样至16x16，构建为LR图像；模型输入尺寸为3x128x128