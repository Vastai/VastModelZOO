## Official

```
link: https://github.com/xinntao/ESRGAN
branch: master
commit: 73e9b634cf987f5996ac2dd33f4050922398a921
```

### step.1 获取预训练模型
克隆原始仓库，将[export.py](./export.py)脚本放置在`{ESRGAN}`目录，执行以下命令，进行模型导出为onnx和torchscript：
```bash
# cd {ESRGAN}
python export.py
```

### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集，通过[image2npz.py](../../common/utils/image2npz.py)，转换为对应npz文件

### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[config.yaml](../vacc_code/build/config.yaml)
3. 执行转换

   ```bash
   vamc build ../vacc_code/build/config.yaml
   ```

### step.4 benchmark

1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path DIV2K/DIV2K_valid_LR_bicubic/X4 \
    --target_path DIV2K/DIV2K_valid_LR_bicubic/X4_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[official-esrgan_x4-vdsp_params.json](../vacc_code/vdsp_params/official-esrgan_x4-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/esrgan_x4-fp16-none-3_128_128-vacc/esrgan_x4 \
    --vdsp_params ../vacc_code/vdsp_params/official-esrgan_x4-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/esrgan_x4-fp16-none-3_128_128-vacc/esrgan_x4 \
    --vdsp_params vacc_code/vdsp_params/official-esrgan_x4-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --gt_dir DIV2K/DIV2K_valid_HR \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/esrgan_x4 \
    --input_shape 128 128 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


#### Tips
- vacc fp16 build需要6小时；int8 build需要5小时
- vacc结果效果不佳，对边缘纹理增强太过明显，可能和resize方式有关