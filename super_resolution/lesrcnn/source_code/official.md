## Official

### step.1 获取预训练模型

```
link: https://github.com/hellloxiaotian/LESRCNN
branch: master
commit: e3729ed3884cbaa67c1534a1ac9626c71d670d27
```
- 拉取原始仓库，修改对应model forward的scale参数，例如修改[x2/model/lesrcnn.py#L82](https://github.com/hellloxiaotian/LESRCNN/blob/master/x2/model/lesrcnn.py#L82)，设置为`scale=2`
- 将[export.py](../source_code/export.py)移动至`{LESRCNN}`目录，修改对应参数，导出torchscript（~onnx暂无法导出，有pixel_shuffle不支持~，opset_version=13 ok）

### step.2 准备数据集
- 下载[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集

- 通过[image2npz.py](../../common/utils/image2npz.py)，将测试低清LR图像转换为对应npz文件


### step.3 模型转换
1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[config.yaml](../vacc_code/build/config.yaml)，执行转换命令：
    ```bash
    vamc build ./vacc_code/build/config.yaml
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
3. 性能测试，配置vdsp参数[official-lesrcnn_x2-vdsp_params.json](../vacc_code/vdsp_params/official-lesrcnn_x2-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/lesrcnn_x2-int8-kl_divergence-3_128_128-vacc/lesrcnn_x2 \
    --vdsp_params ../vacc_code/vdsp_params/official-lesrcnn_x2-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/lesrcnn_x2-int8-kl_divergence-3_128_128-vacc/lesrcnn_x2 \
    --vdsp_params vacc_code/vdsp_params/official-lesrcnn_x2-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,128,128] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --gt_dir DIV2K/DIV2K_valid_HR \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/lesrcnn \
    --input_shape 128 128 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


## Tips
- 由于`pixel_shuffle`的原因，fp16在runmodel和runstream下均无法跑通，会导致系统死机，需重启
- 3x放大，int8 run会报错超时
- 验证版本：`Vaststream 1.1.6 SP13 0322 V2X`
