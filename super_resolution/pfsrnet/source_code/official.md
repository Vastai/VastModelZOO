## Official

```
link: https://github.com/DeokyunKim/Progressive-Face-Super-Resolution
branch: master
commit: 8d7a354fa96c92f6efdb2074732c343a351d2ce8
```

### step.1 获取预训练模型
一般在原始仓库内进行模型转为onnx或torchscript。在原仓库test或val脚本内，如[demo.py#L43](https://github.com/DeokyunKim/Progressive-Face-Super-Resolution/blob/master/demo.py#L43)，定义模型和加载训练权重后，添加以下脚本可实现：
```python

input_shape = (1, 3, 16, 16)
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(generator, input_data).eval()
torch.jit.save(scripted_model, 'pfsrnet.torchscript.pt')

import onnx
torch.onnx.export(generator, input_data, 'pfsrnet.onnx', input_names=["input"], output_names=["output"], opset_version=10,
            # dynamic_axes= {
            #                 "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
            #                 "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}
            #                 }
)
```
> **Note**
>
> 模型还使用两个超参：`step=3, alpha=1`，可在模型定义中[model.py#L109](https://github.com/DeokyunKim/Progressive-Face-Super-Resolution/blob/master/model.py#L109)，将默认`step=1, alpha=-1`，修改为`step=3, alpha=1`
> 

### step.2 准备数据集
- 下载[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)数据集(Align&Cropped Images)，取里面前1k张图像

> **Note**
> 
> 测试图像：即为高清HR图像，测试代码内部会通过渐进式下采样至16尺寸的LR图像，作为模型输入
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
    --dataset_path img_align_celeba_1k \
    --target_path  img_align_celeba_1k_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[official-pfsrnet-vdsp_params.json](../vacc_code/vdsp_params/official-pfsrnet-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/pfsrnet-fp16-none-3_16_16-vacc/pfsrnet \
    --vdsp_params ../vacc_code/vdsp_params/official-pfsrnet-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,16,16]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/pfsrnet-fp16-none-3_16_16-vacc/pfsrnet \
    --vdsp_params vacc_code/vdsp_params/official-pfsrnet-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,16,16] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --gt_dir img_align_celeba_1k \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/pfsrnet \
    --input_shape 16 16 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


#### Tips
- 源仓库没有训练代码，可以借鉴[eval.py](https://github.com/DeokyunKim/Progressive-Face-Super-Resolution/blob/master/eval.py)
- 注意PFSRNet模型需要进行Progressive渐进式预处理（配准，src-178-128-64-32-16-input）才能达到理想精度，直接resize至16*16，效果很差，人脸关键器官会变形，应该和模型内的人脸对齐网络有关
- runstream推理时使用的vdsp算子预处理，只能一步resize，会导致精度很差，所以进入vdsp前现做好渐进式缩放预处理
- 完整评估数据集有278136张，[dataloader.py#L38](https://github.com/DeokyunKim/Progressive-Face-Super-Resolution/blob/master/dataloader.py#L38)，比较大，此处只选择了前1000张，作为评估展示，精度上会有些许差异