
# pytorch

`OCSort`跟踪算法中部署主要是对检测算法`yolox`进行模型转换以及部署，精度测试可以基于`MOT17`数据集在检测算法后接`track`模块进行测试


### step.1 获取预训练模型

基于[ocsort](https://github.com/noahcao/OC_SORT/tree/master)，转换检测模型至onnx格式，将[export_onnx.py](./export_onnx.py)复制到项目中，同时运行如下脚本

```
python tools/export_onnx.py --input input -f exps/example/mot/yolox_x_mot17_ablation_half_train.py  -c models/ocsort_mot17_ablation.pth.tar --output-name ocsort_mot17_ablation.onnx
```


### step.2 准备数据集

- [校准数据集](https://motchallenge.net/data/MOT17/)
- [评估数据集](https://motchallenge.net/data/MOT17/)

### step.3 模型转换

1. 根据具体模型，修改编译配置
    - [pytorch_ocsort.yaml](../build_in/build/pytorch_ocsort.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd ocsort
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/pytorch_ocsort.yaml
    ```

### step.4 模型推理

1. runstream
    - 参考：[ocsort_vsx.py](../build_in/vsx/python/ocsort_vsx.py)
    ```bash
    python ../build_in/vsx/python/ocsort_vsx.py \
        --model_prefix_path deploy_weights/pytorch_ocsort_run_stream_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/pytorch-ocsort_mot17_ablation-vdsp_params.json\
        --path /path/to/mot/MOT17/train \
        --result_dir result/track_eval
    ```

    <details><summary>点击查看精度测试结果</summary>

    ```
    # fp16
                        Rcll  Prcn  GT    MT    PT    ML   FP    FN  IDs   FM  MOTA  MOTP num_objects
    MOT17-02-FRCNN 65.3% 96.0%  62 29.0% 53.2% 17.7% 2.7% 34.7% 0.7% 1.8% 61.9% 0.148       18581
    MOT17-04-FRCNN 86.0% 99.2%  83 63.9% 31.3%  4.8% 0.7% 14.0% 0.1% 0.6% 85.2% 0.101       47557
    MOT17-11-FRCNN 81.7% 97.0%  75 45.3% 30.7% 24.0% 2.5% 18.3% 0.3% 0.8% 78.8% 0.113        9436
    MOT17-10-FRCNN 66.9% 97.8%  57 33.3% 56.1% 10.5% 1.5% 33.1% 1.1% 2.7% 64.3% 0.192       12839
    MOT17-05-FRCNN 72.0% 98.5% 133 28.6% 52.6% 18.8% 1.1% 28.0% 0.8% 1.8% 70.1% 0.148        6917
    MOT17-13-FRCNN 78.5% 98.9% 110 52.7% 35.5% 11.8% 0.8% 21.5% 0.5% 1.4% 77.1% 0.168       11642
    MOT17-09-FRCNN 77.1% 99.9%  26 50.0% 50.0%  0.0% 0.1% 22.9% 0.3% 1.4% 76.7% 0.126        5325
    OVERALL        78.0% 98.4% 546 42.7% 43.2% 14.1% 1.3% 22.0% 0.4% 1.2% 76.3% 0.128      112297
                    IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm num_objects
    MOT17-02-FRCNN 54.8% 67.6% 46.0% 65.3% 96.0%  62  18  33 11  503  6447 123   326 61.9% 0.148  49  65   7       18581
    MOT17-04-FRCNN 86.4% 93.0% 80.6% 86.0% 99.2%  83  53  26  4  338  6651  35   294 85.2% 0.101  17  17   3       47557
    MOT17-11-FRCNN 82.3% 90.0% 75.8% 81.7% 97.0%  75  34  23 18  240  1728  28    79 78.8% 0.113  10  21   3        9436
    MOT17-10-FRCNN 53.3% 65.6% 44.9% 66.9% 97.8%  57  19  32  6  193  4249 146   344 64.3% 0.192  72  80  11       12839
    MOT17-05-FRCNN 71.1% 84.2% 61.5% 72.0% 98.5% 133  38  70 25   77  1936  52   122 70.1% 0.148  52  24  27        6917
    MOT17-13-FRCNN 70.3% 79.4% 63.0% 78.5% 98.9% 110  58  39 13   98  2505  59   165 77.1% 0.168  56  20  18       11642
    MOT17-09-FRCNN 68.7% 78.8% 60.8% 77.1% 99.9%  26  13  13  0    5  1220  18    72 76.7% 0.126  19   5   6        5325
    OVERALL        74.2% 83.9% 66.5% 78.0% 98.4% 546 233 236 77 1454 24736 461  1402 76.3% 0.128 275 232  75      112297

    # int8
                    Rcll  Prcn  GT    MT    PT    ML   FP    FN  IDs   FM  MOTA  MOTP num_objects
    MOT17-02-FRCNN 65.2% 96.0%  62 27.4% 54.8% 17.7% 2.7% 34.8% 0.6% 1.7% 61.8% 0.161       18581
    MOT17-04-FRCNN 85.9% 99.2%  83 65.1% 28.9%  6.0% 0.7% 14.1% 0.1% 0.6% 85.1% 0.116       47557
    MOT17-11-FRCNN 81.8% 96.8%  75 45.3% 33.3% 21.3% 2.7% 18.2% 0.3% 0.9% 78.8% 0.120        9436
    MOT17-10-FRCNN 66.9% 97.8%  57 33.3% 54.4% 12.3% 1.5% 33.1% 1.2% 2.8% 64.1% 0.201       12839
    MOT17-05-FRCNN 71.8% 98.5% 133 29.3% 54.1% 16.5% 1.1% 28.2% 0.8% 1.9% 69.9% 0.151        6917
    MOT17-13-FRCNN 77.5% 98.8% 110 50.9% 38.2% 10.9% 1.0% 22.5% 0.7% 1.7% 75.8% 0.181       11642
    MOT17-09-FRCNN 76.7% 99.9%  26 46.2% 53.8%  0.0% 0.1% 23.3% 0.4% 1.3% 76.2% 0.135        5325
    OVERALL        77.8% 98.3% 546 42.3% 44.3% 13.4% 1.3% 22.2% 0.4% 1.3% 76.0% 0.140      112297
                    IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm num_objects
    MOT17-02-FRCNN 57.0% 70.5% 47.8% 65.2% 96.0%  62  17  34 11  501  6474 120   324 61.8% 0.161  51  63   7       18581
    MOT17-04-FRCNN 88.3% 95.2% 82.4% 85.9% 99.2%  83  54  24  5  320  6712  41   299 85.1% 0.116  20  17   2       47557
    MOT17-11-FRCNN 78.8% 86.1% 72.7% 81.8% 96.8%  75  34  25 16  254  1718  31    82 78.8% 0.120  11  25   5        9436
    MOT17-10-FRCNN 49.6% 61.0% 41.7% 66.9% 97.8%  57  19  31  7  197  4253 154   362 64.1% 0.201  77  84  11       12839
    MOT17-05-FRCNN 68.7% 81.5% 59.4% 71.8% 98.5% 133  39  72 22   76  1954  54   132 69.9% 0.151  56  26  31        6917
    MOT17-13-FRCNN 68.8% 78.2% 61.4% 77.5% 98.8% 110  56  42 12  114  2622  83   197 75.8% 0.181  73  32  25       11642
    MOT17-09-FRCNN 64.3% 74.0% 56.8% 76.7% 99.9%  26  12  14  0    5  1240  20    71 76.2% 0.135  19   5   6        5325
    OVERALL        74.1% 84.0% 66.4% 77.8% 98.3% 546 231 242 73 1467 24973 503  1467 76.0% 0.140 307 252  87      112297
    ```

    </details>


### step.5 性能测试

**Note:** 基于`tracking by detection`的多目标跟踪算法包括检测、track两个流程，modelzoo只提供检测算法模型的性能测试，整体pipeline性能可使用vastpipe测试。精度采用python脚本计算，如上`Step.4`

1. 性能测试
    - 配置[pytorch-ocsort_mot17_ablation-vdsp_params.json](../build_in/vdsp_params/pytorch-ocsort_mot17_ablation-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/pytorch_ocsort_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/pytorch-ocsort_mot17_ablation-vdsp_params.json -i 1 -b 1 -d 0 -p 1
    ```
    