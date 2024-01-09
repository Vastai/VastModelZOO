## Pytorch版本

```
link: https://github.com/jt827859032/DRRN-pytorch
branch: master
commit: 2cb8bc0aecef7503e259e1bb73f95612fd46b3be
```

### step.1 获取预训练模型
一般在原始仓库内进行模型转为onnx或torchscript。在原仓库test或val脚本内，如[eval.py#L30](https://github.com/jt827859032/DRRN-pytorch/blob/master/eval.py#L30)，定义模型和加载训练权重后，添加以下脚本可实现：

```python
model.eval()

input_shape = (1, 1, 256, 256)
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()
torch.jit.save(scripted_model, 'drrn.torchscript.pt')

import onnx
torch.onnx.export(model, input_data, 'drrn.onnx', input_names=["input"], output_names=["output"], opset_version=10,
            # dynamic_axes= {
            #                 "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
            #                 "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}
            #                 }
)
```


### step.2 准备数据集
- 下载[Set5_BMP](https://github.com/twtygqyy/pytorch-vdsr/tree/master/Set5)数据集

### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[config.yaml](../vacc_code/build/config.yaml)
3. 执行转换

   ```bash
   vamc build ../vacc_code/build/config.yaml
   ```

### step.4 benchmark

1. 获取vamp性能测试工具
2. 基于[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`（注意只需要YCrcb颜色空间的Y通道信息）：
    ```bash
    python ../vacc_code/vdsp_params/image2npz.py \
    --dataset_path Set5_BMP/scale_4 \
    --target_path  Set5_BMP/scale_4_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[pytorch-drrn-vdsp_params.json](../vacc_code/vdsp_params/pytorch-drrn-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/drrn-int8-kl_divergence-1_256_256-vacc/drrn \
    --vdsp_params ../vacc_code/vdsp_params/pytorch-drrn-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,256,256]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/drrn-int8-kl_divergence-1_256_256-vacc/drrn \
    --vdsp_params vacc_code/vdsp_params/pytorch-drrn-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,256,256] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --src_dir Set5_BMP/scale_4 \
    --gt_dir Set5_BMP/hr \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/drrn \
    --input_shape 256 256 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


#### Tips
- 注意预处理中的图像缩放resize方式，使用默认cv2.INTER_LINEAR结果图（torch&vacc infer）上会有波纹，使用cv2.INTER_AREA可避免此问题
- 该模型实际只改善图像质量，模型的输出与输入分辨率一致
- 输入为YCrCb颜色空间的Y通道，input尺寸：[1, 256, 256]