### step.1 获取模型
- 首先下载 huggingface 官网下载模型，下载路径：https://huggingface.co/textattack/bert-base-uncased-imdb
  
    ```bash
    cd /path/to/weights
    git clone https://huggingface.co/textattack/bert-base-uncased-imdb
    ```  
- 将 pt 格式模型转换为 torchscript 格式模型， 可参考：[pt2torchscript](./source_code/pretrain_model/README.md)

 
### step.2 获取数据集
由于 compiler 不同版本对 nlp 模型的输入个数有差异，因此用户需根据相应的 compiler 版本进行下载
- compiler v1.5
  - 校准数据集
    - [calib_128](https://drive.google.com/drive/folders/1-OZwjCHLFKeyeClzVCG6Nd6UINFV3AIi)
  - 评估数据集
    - [test_128](https://drive.google.com/drive/folders/1smwetNX7iLPzcTV6fXn8tP-JZjFFtnm4)
- labels： 
    - [eval_labels.txt](https://drive.google.com/drive/folders/1HKT2azvnmc1cUCP2IeZvhSTqFJN6xIx1)

### step.3 模型转换
1. 根据具体模型修改配置文件
    - [huggingface](./build_in/build/bert_imdb.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd bert
    mkdir workspace
    cd workspace
    vamc build ../build_in/build/bert_imdb.yaml
    ```
  
    > 注意：当前配置文件是基于 compiler v1.5 版本的，具体参数配置请用户根据系统、环境版本自行修改

### step.4 模型推理
- 基于 [gen_datalist.py](../../common/utils/gen_datalist.py)，获得对应的 `npz_datalist.txt`

   ```bash
   python ../../common/utils/gen_datalist.py \
       --data_dir /path/to/IMDB/test_128_6input \
       --save_path npz_datalist.txt
   ```

- runstream 运行
  - `compiler version <= 1.5.0 并且 vastsream sdk == 1.X`

    运行 [sample_nlp.py](../../common/sdk1.0/sample_nlp.py) 脚本，获取 runstream 结果，示例：

    ```bash
    cd ../../common/sdk1.0
    python sample_nlp.py \
        --model_info ./network.json \
        --bytes_size 512 \
        --datalist_path npz_datalist.txt \
        --save_dir ./result/dev408
    ```

    > 可参考 [network.json](../../../question_answering/common/sdk1.0/network.json) 进行修改

  - `compiler version >= 1.5.2 并且 vastsream sdk == 2.X`

    运行 [vsx_sc.py](../../common/vsx/python/vsx_sc.py) 脚本，获取 runstream 结果，示例：

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

   基于[imdb_eval.py](../../common/eval/imdb_eval.py)，解析npz结果，并评估精度
   ```bash
   python ../../common/eval/imdb_eval.py --labels_path label.txt --result_dir ./result/imdb_128
   ```

### step.5 性能精度测试
1. 基于[gen_datalist.py](../../common/utils/gen_datalist.py)，获得推理数据`npz`以及对应的`npz_datalist.txt`， 可参考 step.4

2. 执行性能测试：
    ```bash
   vamp -m deploy_weights/bert_base_imdb-int8-max-mutil_input-vacc/bert_base_imdb \
        --vdsp_params ../../common/vamp_info/bert_vdsp.yaml \
        --iterations 1024 \
        --batch_size 1 \
        --instance 6 \
        --processes 2 \
        --datalist npz_datalist.txt \
        --path_output ./save/bert_imdb
    ```
    > 相应的 `vdsp_params` 等配置文件可在 [vamp_info](../../common/vamp_info/) 目录下找到
    >
    > 如果仅测试模型性能可不设置 `datalist`、`path_output` 参数

3. 精度评估
    基于[imdb_eval.py](../../common/eval/imdb_eval.py)，解析和评估npz结果， 可参考 step.4