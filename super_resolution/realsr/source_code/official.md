## Official

```
link: https://github.com/Tencent/Real-SR
branch: master
commit: 10276f4e497894dcbaa6b4244c06bba881b9460c
```

### step.1 获取预训练模型
克隆原始仓库，将[export.py](./export.py)脚本放置在源仓库目录，执行以下命令，进行模型导出为onnx和torchscript。
```bash
# cd {Real-SR/code}
python export.py
```

### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集([Validation Data (HR images)](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip))
- 通过[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，转换为HR和LR及对应npz文件


### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[config.yaml](../vacc_code/build/config.yaml)
3. 执行转换

   ```bash
   vamc build ../vacc_code/build/config.yaml
   ```

### step.4 benchmark

1. 获取vamp性能测试工具
2. 基于[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../vacc_code/vdsp_params/image2npz.py \
    --dataset_path realsr/DIV2K_valid_HR \
    --hr_path realsr/DIV2K_valid_HR_512 \
    --lr_path realsr/DIV2K_valid_LR_128 \
    --target_path realsr/DIV2K_valid_LR_128_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[official-DF2K-vdsp_params.json](../vacc_code/vdsp_params/official-DF2K-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/DF2K-fp16-none-3_128_128-vacc/DF2K \
    --vdsp_params ../vacc_code/vdsp_params/official-DF2K-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/DF2K-fp16-none-3_128_128-vacc/DF2K \
    --vdsp_params vacc_code/vdsp_params/official-DF2K-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --gt_dir realsr/DIV2K_valid_HR_512 \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/DF2K \
    --input_shape 128 128 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


#### Tips
- vacc fp16 build需要3小时；int8 build需要5小时
- 仓库提供了三个权重，基于不同退化图像训练而来，网络结构是一样的，**此处只验证DF2K子模型**
