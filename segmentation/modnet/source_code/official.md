
## official

### step.1 获取预训练模型
```
link: https://github.com/ZHKKKe/MODNet
branch: master
commit: 28165a451e4610c9d77cfdf925a94610bb2810fb
```

- 克隆原始仓库，基于原始仓库代码转换至onnx，[onnx/export_onnx.py](https://github.com/ZHKKKe/MODNet/blob/master/onnx/export_onnx.py)；修改脚本内尺寸信息和权重信息；注意修改此`onnx`文件夹名称为其它，否则可能会误识别成公共库
- 导出后onnx内已包含sigmoid后处理



### step.2 准备数据集
- 下载[PPM-100](https://github.com/ZHKKKe/PPM)数据集，解压

### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[config.yaml](../vacc_code/build/config.yaml)
3. 执行转换

   ```bash
    vamc build ../vacc_code/build/config.yaml
   ```


### step.4 benchmark
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../vacc_code/vdsp_params/image2npz.py \
    --dataset_path datasets/PPM-100/image \
    --target_path  datasets/PPM-100/image_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[official-modnet-vdsp_params.json](../vacc_code/vdsp_params/official-modnet-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/modnet-int8-kl_divergence-3_480_288-vacc/modnet \
    --vdsp_params ../vacc_code/vdsp_params/official-modnet-vdsp_params.json \
    -i 2 p 2 -b 1
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/modnet-int8-kl_divergence-3_480_288-vacc/modnet \
    --vdsp_params vacc_code/vdsp_params/official-modnet-vdsp_params.json \
    -i 2 p 2 -b 1 \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --src_dir datasets/PPM-100/image \
    --gt_dir datasets/PPM-100/matte \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 480 288 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


#### Tips
- vacc int8有掉点，sigma量化方式掉点最小
- 单个评估指标不够准确，可加入分割领域的miou等评估指标，综合评估
- matting评估指标来自于：[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.8/Matting/ppmatting/metrics)