### step.1 模型 finetune
-  huggingface，模型微调说明：[huggingface_mrpc.md](./source_code/finetune/huggingface_mrpc.md)

### step.2 获取模型
- huggingface，预训练模型导出至torchscript格式，说明: [pt2torchscript](./source_code/pretrained_model/README.md)


### step.3 获取数据集
- 校准数据集
    - [huggingface](https://drive.google.com/drive/folders/1FbQr7IYiFJJlY2kCytxYjAkdDLOAJ5Z8)
- [评估数据集-v1.5](https://drive.google.com/drive/folders/1whjFLfxYUjPFOM_ALp17WbfnUTAecDhC)
- labels： [dev.tsv](https://drive.google.com/drive/folders/1kv675JT_IzanhIvB6kiz5_pNFiRdHEFX)
> 注意： 由于 compiler 1.5 版本将 transformer 相关模型的输入改变为6个。 因此，1.5 版本的校验数据集需使用 `评估数据集-v1.5`, 输入长度为512请使用 `评估数据集-v1.5-512`。

### step.4 模型转换
1. 根据具体模型修改配置文件
    - [huggingface](./build_in/build/electra_huggingface.yaml)

    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd electra
    mkdir workspace
    cd workspace
    vamc build ../build_in/build/electra_huggingface.yaml
    ```
   
### step.5 模型推理
- 基于 [sequence2npz.py](../../common/utils/sequence2npz.py)，获得对应的 `npz_datalist.txt`

   ```bash
   python ../../common/utils/sequence2npz.py \
       --npz_path /path/to/MRPC/dev408 \
       --save_path npz_datalist.txt
   ```
- 推理 运行
  - `compiler version <= 1.5.0 并且 vastsream sdk == 1.X`

    运行 [sample_nlp.py](../../common/sdk1.0/sample_nlp.py) 脚本，获取 推理 结果，示例：

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

    运行 [vsx_sc.py](../../common/vsx/python/vsx_sc.py) 脚本，获取 推理 结果，示例：

    ```bash
    cd ../../common/vsx/python/
    python vsx_sc.py \
        --data_list npz_datalist.txt\
        --model_prefix_path ./build_deploy/electra_base_128/electra_base_128 \
        --device_id 0 \
        --batch 1 \
        --save_dir ./result/dev408
    ```

    > `--model_prefix_path` 为转换的模型三件套的文件前缀路径

- 精度评估

   基于[mrpc_eval.py](../../common/eval/mrpc_eval.py)，解析npz结果，并评估精度
   ```bash
   python ../common/eval/mrpc_eval.py --result_dir ./result/dev408 --eval_path /path/to/MRPC/dev.tsv
   ```

### step.6 性能精度测试
1. 基于[sequence2npz.py](../../common/utils/sequence2npz.py)，获得推理数据`npz`以及对应的`npz_datalist.txt`， 可参考 step.5

2. 执行性能测试：
    ```bash
   vamp -m deploy_weights/electra_base_mrpc-int8-max-mutil_input-vacc/electra_base_mrpc \
        --vdsp_params ../../common/vamp_info/bert_vdsp.yaml \
        --iterations 1024 \
        --batch_size 1 \
        --instance 6 \
        --processes 2 \
        --datalist npz_datalist.txt \
        --path_output ./save/electra
    ```
    > 相应的 `vdsp_params` 等配置文件可在 [vamp_info](../../common/vamp_info/) 目录下找到
    >
    > 如果仅测试模型性能可不设置 `datalist`、`path_output` 参数

3. 精度评估
    基于[mrpc_eval.py](../../common/eval/mrpc_eval.py)，解析和评估npz结果， 可参考 step.5

- <details><summary>精度</summary>

    ```bash
    electra_base_gen_mrpc-128 fp16
    F1:  0.8592 Accuracy: 0.8064
    ```

</details>
