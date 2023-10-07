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
1. 获取vamc模型转换工具


2. 根据具体模型修改模型转换配置文件[official-config.yaml](../vacc_code/build/official-config.yaml)，执行转换命令：
    ```bash
    vamc build ./vacc_code/build/official-config.yaml
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
3. 性能测试，配置vdsp参数[official-edsr_x2-vdsp_params.json](../vacc_code/vdsp_params/official-edsr_x2-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/edsr_x2-int8-percentile-3_256_256-vacc/edsr_x2 \
    --vdsp_params ../vacc_code/vdsp_params/official-edsr_x2-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,256,256]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/edsr_x2-int8-percentile-3_256_256-vacc/edsr_x2 \
    --vdsp_params vacc_code/vdsp_params/official-edsr_x2-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,256,256] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [official-vamp_eval.py](../vacc_code/vdsp_params/official-vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/official-vamp_eval.py \
    --gt_dir DIV2K/DIV2K_valid_HR \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/rcan \
    --input_shape 256 256 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


## Tips
- 原始仓库的模型均比较大，在较大尺寸quant时可能会报错：已中断（系统内存问题，需切换至内存较大机器上）
- 无法导出onnx，含pixel_shuffle(opset_version=13 is ok)，只能导出torchscript
- EDSR_x3、EDSR_x4和edsr_baseline_x3，build通过，但run会超时，runtime报内存不足`malloc dlc memory faild, flag:1`
