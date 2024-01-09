## keras

### step.1 获取预训练模型

```
link：https://github.com/zhixuhao/unet
branch: master
commit: b45af4d458437d8281cc218a07fd4380818ece4a
```
- 拉取原始仓库，将[keras](./source_code/keras)文件夹下脚本，替换原始仓库对应文件
- 原始仓库为单通道模型，我们重新训练3通道模型，基于[main.py](./keras/main.py)，保存`h5`含有网络结构和权重



### step.2 准备数据集
- 下载[isbi](https://github.com/zhixuhao/unet/tree/master/data/membrane)数据集


### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[keras-config.yaml](../vacc_code/build/keras-config.yaml)
3. 执行转换

   ```bash
   vamc build ../vacc_code/build/keras-config.yaml
   ```

### step.4 benchmark
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path isbi/train/image \
    --target_path  isbi/train/image_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[keras-unet-vdsp_params.json](../vacc_code/vdsp_params/keras-unet-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/unet-int8-percentile-3_256_256-vacc/unet \
    --vdsp_params ../vacc_code/vdsp_params/keras-unet-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,256,256]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/unet-int8-percentile-3_256_256-vacc/unet \
    --vdsp_params vacc_code/vdsp_params/keras-unet-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,256,256] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [keras-vamp_eval.py](../vacc_code/vdsp_params/keras-vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/keras-vamp_eval.py \
    --src_dir isbi/train/image \
    --gt_dir isbi/train/label \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir output/unet \
    --input_shape 256 256 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```