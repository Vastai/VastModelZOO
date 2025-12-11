### step.1 模型 finetune
-  huggingface，模型微调说明：[huggingface_bert_squad.md](./source_code/finetune/huggingface_bert_squad.md)

### step.2 获取模型
- huggingface，预训练模型导出至torhcscript格式，说明: [pt2torchscript](./source_code/pretrain_model/README.md)

### step.3 获取数据集
- 校准数据集
    - [huggingface](https://drive.google.com/drive/folders/1c2eO8loUa205TxYQe7BYimOA2TJqv7lM)
- [评估数据集-v1.3](https://drive.google.com/drive/folders/1vEbJLCrp-xjzmk31CwMrMy5xeo9DmUGj)
- [评估数据集-v1.5](https://drive.google.com/drive/folders/1TmA2Ck6CM7PNOL_2P75-aRO78y2NJ7DH)

> 注意： 由于 compiler 1.5 版本将 bert 相关模型的输入改变为6个。 因此，1.5 版本的校验数据集需使用 `评估数据集-v1.5`。

### step.4 模型转换
1. 根据具体模型修改配置文件
   - [huggingface_bert_large_en_qa-384](./build_in/build/huggingface_bert_large_en_qa-384.yaml)
   - [huggingface_bert_base_en_qa-384](./build_in/build/huggingface_bert_base_en_qa-384.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd bert
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/huggingface_bert_large_en_qa-384.yaml
    vamc compile ../build_in/build/huggingface_bert_base_en_qa-384.yaml
    ```

### step.5 模型推理
- 基于 [gen_datalist.py](../../common/utils/gen_datalist.py)，获得对应的 `npz_datalist.txt`

   ```bash
   python ../../common/utils/gen_datalist.py \
       --data_dir /path/to/datasets/val_npz_6inputs \
       --save_path npz_datalist.txt
   ```

- runstream 运行
  - `compiler version <= 1.5.0 并且 vastsream sdk == 1.X`

    运行 [sample_nlp.py](../../common/sdk1.0/sample_nlp.py) 脚本，获取 runstream 结果，示例：

    ```bash
    cd ../../common/sdk1.0
    python sample_nlp.py \
        --model_info ./network.json \
        --bytes_size 1536 \
        --datalist_path npz_datalist.txt \
        --save_dir ./output
    ```

    > 可参考 [network.json](../../common/sdk1.0/network.json) 进行修改

  - `compiler version >= 1.5.2 并且 vastsream sdk == 2.X`

    运行 [vsx_sc.py](../../common/vsx/python/vsx_qa.py) 脚本，获取 runstream 结果，示例：

    ```bash
    python ../../common/vsx/python/vsx_qa.py \
        --data_list npz_datalist.txt\
        --model_prefix_path ./deploy_weights/bert_base_en_qa-384/mod \
        --device_id 0 \
        --batch 1 \
        --save_dir ./bert_base_en_qa-384_out

    python ../../common/vsx/python/vsx_qa.py \
        --data_list npz_datalist.txt\
        --model_prefix_path ./deploy_weights/bert_large_en_qa-384/mod \
        --device_id 0 \
        --batch 1 \
        --save_dir ./bert_large_en_qa-384_out
    ```

    > `--model_prefix_path` 为转换的模型三件套的文件前缀路径

- 精度评估

   基于[squad_eval.py](../../common/eval/squad_eval.py)，解析npz结果，并评估精度
   ```bash
   python ../../common/eval/squad_eval.py  \
       --result_dir ./bert_base_en_qa-384_out\
       --eval_path /path/to/dev-v1.1.json \
       --vocab_path ../../common/vocab.txt

   python ../../common/eval/squad_eval.py  \
       --result_dir ./bert_large_en_qa-384_out\
       --eval_path /path/to/dev-v1.1.json \
       --vocab_path ../../common/vocab.txt
   ```
    > [dev-v1.1.json](https://drive.google.com/drive/folders/1M37nwzZL5C606k4El556lUkBsIETvxIT)

### step.6 性能精度测试
1. 基于[sequence2npz.py](../../common/utils/sequence2npz.py)，获得推理数据`npz`以及对应的`npz_datalist.txt`, 可参考 step 5

2. 获得模型性能信息：

    ```bash
   vamp -m deploy_weights/bert_base_squad-int8-mse-mutil_input-vacc/bert_base_squad \
        --vdsp_params ../../common/vamp_info/bert_vdsp.json \
        --iterations 1024 \
        --batch_size 1 \
        --instance 6 \
        --processes 2
    ```

3. 执行测试：

    ```bash
    vamp -m deploy_weights/bert_base_squad-int8-mse-mutil_input-vacc/bert_base_squad \
        --vdsp_params ../../common/vamp_info/bert_vdsp.json \
        --hwconfig ../../common/vamp_info/bert_hw_config.bin \
        --batch_size 1 \
        --instance 6 \
        --processes 2
        --datalist npz_datalist.txt \
        --path_output ./save/bert
    ```

    > 相应的 `vdsp_params`  可在 [vamp_info](../../common/vamp_info/) 目录下找到
    >
    > 如果仅测试模型性能可不设置 `datalist`、`path_output` 参数
    
4. 精度评估，可参考 step.5