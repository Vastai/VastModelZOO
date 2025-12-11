 ### step.1 获取模型
下载 CIB Electra onnx 结构模型，下载路径：[electra_small-512.onnx](https://drive.google.com/drive/folders/1ii0Kz6nxZujiMkoMozrWLbBCGpjmWqh2?usp=sharing)
  
### step.2 获取数据集
由于 compiler 不同版本对 nlp 模型的输入个数有差异，因此用户需根据相应的 compiler 版本进行下载
- compiler v1.5
  - 校准数据集
    - [calib_512](https://drive.google.com/drive/folders/1eWMsZwq4YXKuA1yuyqCKdNYHDVgf7-SX)
  - 评估数据集
    - [test_512](https://drive.google.com/drive/folders/1-0XdQyKbbYHbXpAG6m25CKqllB_MMumc)
- labels： 
    - [label.txt](https://drive.google.com/drive/folders/1oBBbs1vFBfiwgzkw5HyfYCRmAcSwAgXB)

### step.3 模型转换
1. 根据具体模型修改配置文件
  - [CIB](./build_in/build/electra_cib.yaml)
   
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd electra
    mkdir workspace
    cd workspace
    vamc build ../build_in/build/electra_cib.yaml
    ```
    > 注意：当前配置文件是基于 compiler v1.5 版本的，具体参数配置请用户根据系统、环境版本自行修改

### step.4 模型推理

- 基于 [gen_datalist.py](../../common/utils/gen_datalist.py)，获得对应的 `npz_datalist.txt`

   ```bash
   python ../../common/utils/gen_datalist.py \
       --data_dir ./datasets/civ/test_512_6input \
       --save_path npz_datalist.txt
   ```

- 推理 运行
  - `compiler version <= 1.5.0 并且 vastsream sdk == 1.X`

    运行 [sample_nlp.py](../../common/sdk1.0/sample_nlp.py) 脚本，获取 推理 结果，示例：

    ```bash
    cd ../../common/sdk1.0/
    python sample_nlp.py \
        --model_info ./network.json \
        --bytes_size 512 \
        --datalist_path npz_datalist.txt \
        --save_dir ./result/dev408
    ```

    > 可参考 [network.json](../../../question_answering/common/sdk1.0/network.json) 进行修改

  - `compiler version >= 1.5.2 并且 vastsream sdk == 2.X`

    运行 [vsx_sc.py](../../common/vsx/python/vsx_sc.py) 脚本，获取 推理 结果，示例：

    ```bash
    cd ../../common/vsx/python/
    python vsx_sc.py \
        --data_list npz_datalist.txt\
        --model_prefix_path ./build_deploy/bert_base_128/bert_base_128 \
        --device_id 0 \
        --batch 1 \
        --save_dir ./result/dev408
    ```

    > `--model_prefix_path` 为转换的模型三件套的文件前缀路径

- 精度评估

   基于[cib_electra_eval.py](../../common/eval/cib_electra_eval.py)，解析npz结果，并评估精度
   ```bash
   python ../../common/eval/cib_electra_eval.py --labels_path label.txt --result_dir ./result/cib_512
   ```


### step.5 性能精度测试
1. 基于[gen_datalist.py](../../common/utils/gen_datalist.py)，获得推理数据`npz`以及对应的`npz_datalist.txt`， 可参考 step.4

2. 执行性能测试：
    ```bash
   vamp -m deploy_weights/electra_cib_512/electra_cib_512 \
        --vdsp_params ../../common/vamp_info/bert_vdsp.yaml \
        --iterations 1024 \
        --batch_size 1 \
        --instance 6 \
        --processes 2 \
        --datalist npz_datalist.txt \
        --path_output ./save/electra_cib_512
    ```
    > 相应的 `vdsp_params` 等配置文件可在 [vamp_info](../../common/vamp_info/) 目录下找到
    >
    > 如果仅测试模型性能可不设置 `datalist`、`path_output` 参数

3. 精度评估

    基于[cib_electra_eval.py](../../common/eval/cib_electra_eval.py)，，解析和评估npz结果， 可参考 step.4