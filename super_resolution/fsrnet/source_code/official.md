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
> onnx，将原始模型转为onnx时会报错，[OP#14691](http://openproject.vastai.com/projects/model-debug/work_packages/14691/activity)，（TVM output_padding为1x1不支持）修改后重新训练，精度较差。所以不建议转为onnx
> 
> torchscript，可正常转换，但在vacc run结果中会丢失一个像素（TVM reflection_pad2d 不支持）
> 

### step.2 准备数据集
- 下载[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)数据集，取test_img
- 处理好的数据集
  - 测试图像：[FSRNet/test_img](http://10.23.4.235:8080/datasets/sr/CelebAMask-HQ/FSRNet/test_img/?download=zip)
  - 测试图像npz：[FSRNet/test_img_npz](http://10.23.4.235:8080/datasets/sr/CelebAMask-HQ/FSRNet/test_img_npz/?download=zip)
  - 测试图像npz_datalist.txt：[npz_datalist.txt](http://10.23.4.235:8080/datasets/sr/CelebAMask-HQ/FSRNet/npz_datalist.txt)

> **Note**
> 
> 测试图像：即为高清HR图像，测试代码内部会通过渐进式下采样至128尺寸的LR图像，作为模型输入
> 
> 测试图像npz：通过[image2npz.py](../vacc_code/vdsp_params/image2npz.py)生成，已转换至LR图像

### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[config.yaml](../vacc_code/build/config.yaml)
3. 执行转换

   ```bash
   vamc build ../vacc_code/build/config.yaml
   ```

### step.4 benchmark

1. 获取vamp性能测试工具
2. 基于[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`（注意，HR图像采用渐进式下采样至LR图像）：
    ```bash
    python ../vacc_code/vdsp_params/image2npz.py \
    --dataset_path FSRNet/test_img \
    --target_path  FSRNet/test_img_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[official-fsrnet-vdsp_params.json](../vacc_code/vdsp_params/official-fsrnet-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/fsrnet-fp16-none-3_128_128-vacc/fsrnet \
    --vdsp_params ../vacc_code/vdsp_params/official-fsrnet-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/fsrnet-fp16-none-3_128_128-vacc/fsrnet \
    --vdsp_params vacc_code/vdsp_params/official-fsrnet-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --gt_dir FSRNet/test_img \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/fsrnet \
    --input_shape 128 128 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


#### Tips
- 注意，依照源代码预处理时需要用[clip](https://github.com/cs-giung/FSRNet-pytorch/blob/master/utils.py#L16)，截取范围至[-1, 1]

- 仓库未提供FSRGAN权重，所以暂未验证
