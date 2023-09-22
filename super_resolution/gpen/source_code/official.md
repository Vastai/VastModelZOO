### step.1 获取预训练模型
参考：[export.md](./export.md)

### step.2 准备数据集
- 按论文，取[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)数据集的前1000张图像作为HQ
- 基于[hq2lq.py](./source_code/hq2lq.py)，使用退化模型将高清图像转换为低质图像
- 基于[image2npz.py](../../common/utils/image2npz.py)，将低清LQ图像转为npz格式


### step.3 模型转换
1. 获取vamc模型转换工具


2. 根据具体模型修改模型转换配置文件[config.yaml](../vacc_code/build/config.yaml)，执行转换命令：
    ```bash
    vamc build ../vacc_code/build/config.yaml
    ```
### step.4 benchmark

1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path GPEN/lq \
    --target_path GPEN/lq_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[official-gpen-vdsp_params.json](../vacc_code/vdsp_params/official-gpen-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/gpen-int8-mse-3_512_512-vacc/gpen \
    --vdsp_params ../vacc_code/vdsp_params/official-gpen-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,512,512]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/gpen-int8-mse-3_512_512-vacc/gpen \
    --vdsp_params vacc_code/vdsp_params/official-gpen-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,512,512] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --gt_dir GPEN/hq \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/gpen \
    --input_shape 512 512 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```

## Tips
- GPEN内部定义为：FaceSR
- build_config参数中的data_type需为0（设为0：psnr 25，设为2：psnr 18），在vamc >=2.2版本以上支持
- onnx无法build