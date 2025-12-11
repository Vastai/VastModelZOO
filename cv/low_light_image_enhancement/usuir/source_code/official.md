## Official

### step.1 获取预训练模型

```
link: https://github.com/zhenqifu/USUIR
branch: main
commit: 33683d54bb0c7617e584546257f1bdd73630aa41
```
- 原始模型在forward时返回两个参数，只需要第一个，修改[net/net.py#L14](https://github.com/zhenqifu/USUIR/blob/main/net/net.py#L14)，为`return x_j`
- 参考[export.py](../source_code/official/export.py)，将此文件移动至原始仓库，并修改原始权重路径，导出torchscript和onnx


### step.2 准备数据集
- 下载[UIEB & RUIE](https://drive.google.com/file/d/1DBCXCa5GWJPB7S6xO7f0N562FqXhsV6c/view?usp=sharing)数据集
- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低照度LR图像转换为对应npz文件


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_usuir.yaml](../build_in/build/official_usuir.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd usuir
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_usuir.yaml
    ```

### step.4 模型推理

    - 参考：[official_vsx_inference.py](../build_in/vsx/python/official_vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/official_vsx_inference.py \
        --lr_image_dir  /path/to/UIE/UIEBD/test/image \
        --hr_image_dir /path/to/UIE/UIEBD/test/label \
        --model_prefix_path deploy_weights/official_usuir_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-usuir_uieb-vdsp_params.json \
        --save_dir ./infer_output \
        --device 0
    ```

    ```
    # fp16
    mean psnr: 20.48552681262701, mean ssim: 0.8551101116435483

    # int8
    mean psnr: 20.437147993027697, mean ssim: 0.8425831053200421
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[official-usuir_uieb-vdsp_params.json](../build_in/vdsp_params/official-usuir_uieb-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/official_usuir_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-usuir_uieb-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,720,1280]
    ```

2. 精度测试
    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式（注意原图片格式为`.jpg`），生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path UIE-dataset/UIEBD/test/image \
        --target_path  UIE-dataset/UIEBD/test/image_npz \
        --text_path npz_datalist.txt
    ```
    
    - vamp推理得到npz结果
    ```bash
    vamp -m deploy_weights/official_usuir_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-usuir_uieb-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,720,1280] \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[official-vamp_eval.py](../build_in/vdsp_params/official-vamp_eval.py)
   ```bash
    python ../build_in/vdsp_params/official-vamp_eval.py \
        --gt_dir UIE-dataset/UIEBD/test/label \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output/ \
        --input_shape 720 1280 \
        --draw_dir npz_draw_result \
        --vamp_flag
   ```
