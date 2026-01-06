## Official版本

```
link: https://github.com/cs-giung/FSRNet-pytorch
branch: master
commit: 5b67fdf0657e454d1b382faafbeaf497560f4dc0
```

### step.1 获取预训练模型
一般在原始仓库内进行模型转为onnx或torchscript。在原仓库test或val脚本内，如[demo.py#L43](https://github.com/cs-giung/FSRNet-pytorch/blob/master/demo.py#L43)，定义模型和加载训练权重后，添加以下脚本可实现：
```python
net.eval()

input_shape = (1, 3, 128, 128)
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(net, input_data).eval()
torch.jit.save(scripted_model, 'fsrnet.torchscript.pt')

import onnx
torch.onnx.export(net, input_data, 'fsrnet.onnx', input_names=["input"], output_names=["output"], opset_version=10,
            # dynamic_axes= {
            #                 "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
            #                 "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}
            #                 }
)
```
> **Note**
> 
> 将原始模型转为onnx时会报错，所以不建议转为onnx; torchscript，可正常转换
> 


### step.2 准备数据集
- 下载[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)数据集，取test_img

> **Note**
> 
> 测试图像：即为高清HR图像，测试代码内部会通过渐进式下采样至128尺寸的LR图像，作为模型输入
> 


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_fsrnet.yaml](../build_in/build/official_fsrnet.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd fsrnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_fsrnet.yaml
    ```

### step.4 模型推理

- 参考[vsx_inference](../build_in/vsx/python/vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/vsx_inference.py \
        --lr_image_dir  /path/to/CelebAMask-HQ/test_img \
        --model_prefix_path deploy_weights/official_fsrnet_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-fsrnet-vdsp_params.json \
        --hr_image_dir /path/to/CelebAMask-HQ/test_img \
        --save_dir ./infer_output \
        --device 0
    ```
    
    <details><summary>点击查看精度测试结果</summary>

    ```
    # fp16
    mean psnr:17.342268117430805, mean ssim:0.5047064993885868

    # int8 
    mean psnr:18.043580864450636, mean ssim:0.52853427406625
    ```

    </details>

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[official-fsrnet-vdsp_params.json](../build_in/vdsp_params/official-fsrnet-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_fsrnet_int8/mod \
    --vdsp_params ../build_in/vdsp_params/official-fsrnet-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128]
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，基于[image2npz.py](../build_in/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`（注意，HR图像采用渐进式下采样至LR图像）：
    ```bash
    python ../build_in/vdsp_params/image2npz.py \
        --dataset_path /path/to/FSRNet/test_img \
        --target_path  /path/to/FSRNet/test_img_npz \
        --text_path npz_datalist.txt
    ```
    
    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/official_fsrnet_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-fsrnet-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,128,128] \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)，
    ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
        --gt_dir /path/to/FSRNet/test_img \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output \
        --input_shape 128 128 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```
