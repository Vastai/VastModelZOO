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
1. 根据具体模型，修改编译配置
    - [official_drrn.yaml](../build_in/build/official_drrn.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd drrn
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_drrn.yaml
    ```

### step.4 模型推理

- 参考[vsx_inference.py](../build_in/vsx/python/vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/vsx_inference.py \
        --lr_image_dir  /path/to/Set5_BMP/scale_4 \
        --model_prefix_path deploy_weights/official_drrn_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/pytorch-drrn-vdsp_params.json \
        --hr_image_dir /path/to/Set5_BMP/hr \
        --save_dir ./infer_output \
        --device 0
    ```

    ```
    # fp16
    mean psnr: 30.961503608889927

    # int8 
    mean psnr: 30.407957712286855
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[pytorch-drrn-vdsp_params.json](../build_in/vdsp_params/pytorch-drrn-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_drrn_int8/mod \
        --vdsp_params ../build_in/vdsp_params/pytorch-drrn-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,256,256]
    ```

2. 精度测试
    - 数据准备，基于[image2npz.py](../build_in/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`（注意只需要YCrcb颜色空间的Y通道信息）：
    ```bash
    python ../build_in/vdsp_params/image2npz.py \
    --dataset_path /path/to/Set5_BMP/scale_4 \
    --target_path  /path/to/Set5_BMP/scale_4_npz \
    --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/official_drrn_int8/mod \
        --vdsp_params ../build_in/vdsp_params/pytorch-drrn-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,256,256] \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)
    ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
        --src_dir /path/to/Set5_BMP/scale_4 \
        --gt_dir /path/to/Set5_BMP/hr \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output \
        --input_shape 256 256 \
        --draw_dir npz_draw_result \
        --vamp_flag
   ```
