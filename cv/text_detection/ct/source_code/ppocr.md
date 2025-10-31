## PPOCR

```
link: https://github.com/PaddlePaddle/PaddleOCR
branch: v2.7
commit: b17c2f3a5687186caca590a343556355faacb243
```

### step.1 获取预训练模型
首先，ppocr下载的是训练模型，需要转换为推理模型。在ppocr仓库目录内，执行：

```shell
python3 tools/export_model.py -c configs/det/det_r18_vd_ct.yml  -o Global.pretrained_model=./models/detection/CT/det_r18_ct_train/best_accuracy  Global.save_inference_dir=./models/detection/CT/inference
```

然后，将推理模型转换为onnx：

```shell
paddle2onnx --model_dir ./models/detection/CT/inference --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./models/detection/CT/det_r18_vd_ct.onnx --opset_version 10
```

### step.2 准备数据集
- 下载[Total Text](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset)数据集，通过[image2npz.py](../../common/utils/image2npz.py)，转换为对应npz文件


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [ppocr_ct.yaml](../build_in/build/ppocr_ct.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd ct
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/ppocr_ct.yaml
    ```

### step.4 模型推理
1. runstream
    - 参考：[ct_vsx.py](../build_in/vsx/python/ct_vsx.py)
    ```bash
    python ../build_in/vsx/python/ct_vsx.py \
        --file_path  /path/to/total_text_ppocr/test/rgb/  \
        --model_prefix_path deploy_weights/ppocr_ct_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/ppocr-det_r18_vd_ct-vdsp_params.json \
        --label_txt /path/to/total_text_ppocr/test/test_ppocr.txt \
        --device 0
    ```

    <details><summary>点击查看精度测试结果</summary>

    ```
    # fp16
    metric:  
    {
        'total_num_gt': 2215, 
        'total_num_det': 2094, 
        'global_accumulative_recall': 1622.9999999999966, 
        'hit_str_count': 0, 
        'recall': 0.7327313769751678, 
        'precision': 0.7578796561604569, 
        'f_score': 0.7450933767365878, 
        'seqerr': 1.0, 
        'recall_e2e': 0.0, 
        'precision_e2e': 0.0, 
        'f_score_e2e': 0
    }

    # int8
    metric:  
    {
        'total_num_gt': 2215, 
        'total_num_det': 2078, 
        'global_accumulative_recall': 1623.599999999996, 
        'hit_str_count': 0, 
        'recall': 0.7330022573363413, 
        'precision': 0.7543792107795942, 
        'f_score': 0.7435371170645095, 
        'seqerr': 1.0, 
        'recall_e2e': 0.0, 
        'precision_e2e': 0.0, 
        'f_score_e2e': 0
    }
    ```

    </details>

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[ppocr-det_r18_vd_ct-vdsp_params.json](../build_in/vdsp_params/ppocr-det_r18_vd_ct-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/ppocr_ct_run_stream_int8/mod \
        --vdsp_params ../build_in/vdsp_params/ppocr-det_r18_vd_ct-vdsp_params.json \
        -i 1 p 1 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path /path/to/test/rgb \
        --target_path /path/to/test/rgb_npz \
        --text_path npz_datalist.txt
    ```

    
    - vamp推理得到npz结果
    ```bash
    vamp -m deploy_weights/ppocr_ct_run_stream_int8/mod \
        --vdsp_params ../build_in/vdsp_params/ppocr-det_r18_vd_ct-vdsp_params.json \
        -i 1 p 1 -b 1 \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)，
   ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
        --gt_dir /path/to/icdar2015/Challenge4/ch4_test_images \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output \
        --input_shape 640 960 \
        --draw_dir npz_draw_result \
        --vamp_flag
   ```
