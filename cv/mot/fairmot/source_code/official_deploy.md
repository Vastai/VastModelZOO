# official

### step.1 获取预训练模型

基于[FairMOT](https://github.com/ifzhang/FairMOT)修改代码生成onnx格式模型文件

1. 在`$FairMOT/src/lib/tracker/multitracker.py`中`JDETracker`类`update`函数中添加如下代码
    ```python
    torch_out = torch.onnx._export(self.model, im_blob, 'fairmot_yolov5s.onnx', export_params=True, verbose=False,
                                    input_names=input_names, output_names=output_names, opset_version=11)
    ```

2. 运行如下脚本转换模型至onnx格式
    >模型权重`fairmot_lite.pth`下载参考[FairMOT](https://github.com/ifzhang/FairMOT/blob/4aa62976bde6266cbafd0509e24c3d98a7d0899f/README.md#training)

    ```
    python track.py mot --load_model ../model/fairmot_lite.pth --conf_thres 0.6 --arch yolo --gpus -1
    ```


### step.2 准备数据集

- [校准数据集](https://motchallenge.net/data/MOT16/)
- [评估数据集](https://motchallenge.net/data/MOT16/)

### step.3 模型转换

1. 根据具体模型，修改编译配置
    - [official_fairmot.yaml](../build_in/build/official_fairmot.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd fairmot
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_fairmot.yaml
    ```

### step.4 模型推理

- 参考：[fairmot_vsx.py](../build_in/vsx/python/fairmot_vsx.py)
    ```bash
    python ../build_in/vsx/python/fairmot_vsx.py \
        --image_dir  /media/vastml/dataset/mot/MOT17/train  \
        --model_weight_path deploy_weights/official_fairmot_int8/ \
        --model_name mod \
        --vdsp_params_info ../build_in/vdsp_params/official-fairmot_yolov5s-vdsp_params.json \
        --save_dir ./infer_output 
    ```

    <details><summary>点击查看精度测试结果</summary>

    ```
    # fp16
                  IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm
    MOT17-02-SDP 48.6% 60.2% 40.7% 62.9% 93.0%  62  14  36 12  878  6900 219   617 57.0% 0.214 140  48  20
    MOT17-04-SDP 82.2% 85.6% 79.0% 89.8% 97.3%  83  68  11  4 1178  4848 118   561 87.1% 0.182  36  55   6
    MOT17-05-SDP 64.4% 76.0% 55.8% 70.0% 95.3% 133  37  76 20  239  2074  97   216 65.2% 0.182  89  37  40
    MOT17-09-SDP 58.8% 66.2% 52.9% 78.0% 97.7%  26  16  10  0   98  1171  55   100 75.1% 0.171  36  16   6
    MOT17-10-SDP 63.4% 72.8% 56.2% 72.5% 94.0%  57  24  32  1  596  3526 158   532 66.7% 0.233 102  40  16
    MOT17-11-SDP 65.7% 71.5% 60.8% 81.4% 95.7%  75  38  28  9  347  1759  75   153 76.9% 0.158  47  27  14
    MOT17-13-SDP 66.5% 74.7% 59.9% 73.9% 92.1% 110  47  53 10  737  3040 114   503 66.6% 0.235  90  48  44
    OVERALL      69.9% 77.1% 63.9% 79.2% 95.6% 546 244 246 56 4073 23318 836  2682 74.9% 0.194 540 271 146

    # int8
                  IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm
    MOT17-02-SDP 46.9% 67.2% 36.0% 52.0% 97.3%  62  10  35 17  271  8918 214   584 49.4% 0.235 135  59  21
    MOT17-04-SDP 76.1% 82.2% 70.9% 85.6% 99.2%  83  58  20  5  315  6841 154   924 84.6% 0.196  60  78   9
    MOT17-05-SDP 54.0% 61.6% 48.1% 59.9% 76.7% 133  28  76 29 1258  2768 112   283 40.1% 0.188  76  55  35
    MOT17-09-SDP 53.7% 61.0% 48.0% 74.5% 94.7%  26  13  12  1  224  1358  71   144 69.0% 0.186  36  33   5
    MOT17-10-SDP 62.3% 76.6% 52.4% 65.3% 95.4%  57  14  39  4  405  4454 122   668 61.2% 0.245  82  39  17
    MOT17-11-SDP 64.2% 73.6% 57.0% 73.2% 94.6%  75  23  36 16  397  2528  59   257 68.4% 0.169  41  21  14
    MOT17-13-SDP 63.3% 78.5% 53.0% 64.3% 95.3% 110  31  56 23  372  4157 151   731 59.8% 0.250 108  33  42
    OVERALL      65.6% 76.4% 57.5% 72.4% 96.2% 546 177 274 95 3242 31024 883  3591 68.7% 0.207 538 318 143
    ```

    </details>


### step.5 性能测试

**Note:** JDE多目标跟踪算法包括检测、track两个流程，modelzoo只提供算法模型的性能测试，整体pipeline性能可使用vastpipe测试。精度采用python脚本计算，如上`Step.4`

1. 性能测试
    - 配置[official-fairmot_yolov5s-vdsp_params.json](../build_in/vdsp_params/official-fairmot_yolov5s-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_fairmot_int8/mod --vdsp_params ../build_in/vdsp_params/official-fairmot_yolov5s-vdsp_params.json -i 1 -b 1 -d 0 -p 1
    ```
