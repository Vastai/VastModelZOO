## Official

### step.1 获取预训练模型

```
link: https://github.com/sanghyun-son/EDSR-PyTorch
branch: master
commit: 8dba5581a7502b92de9641eb431130d6c8ca5d7f
```
- 拉取原始仓库，将[export.py](./official/export.py)移动至`{EDSR-PyTorch/src}`目录下，参考[src/demo.sh](https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/demo.sh)，修改对应[src/option.py](https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/option.py)，配置`--scale`和`--pre_train`参数
- 执行转换：
    ```bash
    cd {EDSR-PyTorch}
    python src/export.py
    ```

### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集
- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_edsr.yaml](../build_in/build/official_edsr.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    
    ```bash
    cd edsr
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_edsr.yaml
    ```

### step.4 模型推理
1. runstream
    - 参考[official_vsx_inference.py](../build_in/vsx/python/official_vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/official_vsx_inference.py \
        --lr_image_dir  /path/to/DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --model_prefix_path deploy_weights/official_edsr_run_stream_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-edsr_x2-vdsp_params.json \
        --hr_image_dir /path/to/DIV2K/DIV2K_valid_HR \
        --save_dir ./runstream_output \
        --device 0
    ```

    - 测试精度在打印信息中，如下：
    ```
    # fp16
    mean psnr: 22.297567553171646, mean ssim: 0.742541428324517

    # int8 
    mean psnr: 22.290764919954253, mean ssim: 0.7375210465577184
    ```

### step.5 性能精度测试
1. 性能测试，
    - 配置vdsp参数[official-edsr_x2-vdsp_params.json](../build_in/vdsp_params/official-edsr_x2-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_edsr_run_stream_int8/mod \
    --vdsp_params ../build_in/vdsp_params/official-edsr_x2-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,256,256]
    ```
    
2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path DIV2K/DIV2K_valid_LR_bicubic/X2 \
    --target_path  DIV2K/DIV2K_valid_LR_bicubic/X2_npz \
    --text_path npz_datalist.txt
    ```

    - vamp推理，获得npz结果
    ```bash
    vamp -m deploy_weights/official_edsr_run_stream_int8/mod \
    --vdsp_params ../build_in/vdsp_params/official-edsr_x2-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,256,256] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```

    - 解析npz结果，统计精度：[official-vamp_eval.py](../build_in/vdsp_params/official-vamp_eval.py)
    ```bash
    python ../build_in/vdsp_params/official-vamp_eval.py \
    --gt_dir DIV2K/DIV2K_valid_HR \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir npz_output \
    --input_shape 256 256 \
    --draw_dir npz_draw_result \
    --vamp_flag
    ```
