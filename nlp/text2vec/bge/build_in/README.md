# Build_In Deploy

## step.1 模型准备

1. 下载模型权重

- Embedding
    |                          Model Name                          | Dimension | Sequence Length |                         Introduction                         |
    | :----------------------------------------------------------: | :-------: | :-------------: | :----------------------------------------------------------: |
    |      [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)       |   1024    |      8192       | multilingual; unified fine-tuning (dense, sparse, and colbert) </br> from bge-m3-unsupervised |
    | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |    384    |       512       |                      English model                         |
    | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |    768    |       512       |                        English model                         |
    | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |   1024    |       512       |                      English model                         |
    | [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) |    384    |       512       |                      Chinese model
    | [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |    768    |       512       |                        Chinese model                         |
    | [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) |   1024    |       512       |                      Chinese model                         |
    
    >  base on XLMRobertaModel

- ReRanker

    | 模型 |   基础模型  |   语言   | Dimension | Sequence Length  | Note                                                         |
    | :----------------------------------------------------------: | :----------------------------------------------------------: | :------: | :--: |:--: |:--: | 
    | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)  |  中英文  |  768 | 512 | 轻量级重排序模型，易于部署，推理速度快。                     |
    | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | [xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large) |  中英文  |   1024 | 512 | 轻量级重排序模型，易于部署，推理速度快。                     |
    | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |         [bge-m3](https://huggingface.co/BAAI/bge-m3)         | 多种语言 |   1024 | 8192 | 轻量级重排序模型，具有强大的多语言能力，易于部署，推理速度快。 |


2. 模型导出onnx：[onnx_export.py](../../common/source_code/onnx_export.py)

    ```bash
    python ../../common/source_code/onnx_export.py \
        --model bge/bge-m3 \
        --type embedding \
        --seqlen 512 \
        --save_dir ./onnx_weights
    ```

    **Note:** 若模型文件超过2G，则转换脚本中可在`onnx.save`添加配置`save_as_external_data=True, all_tensors_to_one_file=True`

### step.2 数据集
1. 精度评估数据集：
    - embedding
        - 英文：[mteb/sts12-sts](https://huggingface.co/datasets/mteb/sts12-sts)
        - 中文：[C-MTEB/BQ](https://huggingface.co/datasets/C-MTEB/BQ)
    - reranker：[zyznull/msmarco-passage-ranking](https://huggingface.co/datasets/zyznull/msmarco-passage-ranking)
    - 数据集下载和转换为jsonl格式：[download_datasets.py](../../common/source_code/download_datasets.py)
2. 量化数据集：
    - [gen_quant_data.py](../../common/source_code/gen_quant_data.py)，基于以上数据集，指定seqlen，合成npz量化数据集

### step.3 模型转换
1. 根据具体模型修改模型转换配置文件
    - [embedding_config_fp16.yaml](./build/embedding_config_fp16.yaml)
    - [embedding_config_int8.yaml](./build/embedding_config_int8.yaml)
    - [reranker_config_fp16.yaml](./build/reranker_config_fp16.yaml)
    - [reranker_config_int8.yaml](./build/reranker_config_int8.yaml)

    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd bge
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/embedding_config_fp16.yaml
    ```

### step.4 模型推理
1. 推理：[demo.py](./vsx/demo.py)
    - 配置模型路径等参数，推理脚本内指定的文本对

    ```bash
    python ../build_in/vsx/demo.py \
        --vacc_weight ./vacc_deploy/bge-m3-512-fp16/mod \
        --torch_weight /path/to/bge/bge-m3 \
        --task embedding \
        --eval_engine vacc \
        --eval_dataset /path/to/mteb-sts12-sts_test.jsonl \
        --seqlen 512
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数：[embedding-vdsp_params.json](./vdsp_params/embedding-vdsp_params.json)

    ```bash
    vamp -m vacc_deploy/bge-m3-512-fp16/mod \
    --vdsp_params ../build_in/vdsp_params/embedding-vdsp_params.json  \
    -i 1 p 1 -b 1 -s [[1,512],[1,512],[1,512],[1,512],[1,512],[1,512]] --dtype uint32
    ```

2. 精度测试：[demo.py](./vsx/demo.py)
    - 配置模型路径等参数，指定`--eval_mode`参数为True，进行精度评估

    ```bash
    python ../build_in/vsx/demo.py \
        --vacc_weight ./vacc_deploy/bge-m3-512-fp16/mod \
        --torch_weight /path/to/bge/bge-m3 \
        --task embedding \
        --eval_mode \
        --eval_engine vacc \
        --eval_dataset /path/to/mteb-sts12-sts_test.jsonl \
        --seqlen 512
    ```

### Tips
- reranker模型，不需要指定`output_layout`编译参数
- 注意模型本身只需3个输入，但编译器需要6个输入
- bge-m3模型，int8量化精度掉点验证，可使用以下量化参数实现混合精度量化，跳过一些层保留fp16
    ```yaml
    quantize:
        calibrate_mode: percentile
        quantize_per_channel: false
        overflow_adaptive: 1
        weight_scale: max
        calibrate_chunk_by: -1
        exclude_layers: [1, 2, 5, 6, 10, 11, 12, 15, 16, 19, 20, 22, 26, 27, 28, 31, 32, 35, 36, 38, 42, 43, 44, 47, 48, 51, 52, 54, 58, 59, 60, 63, 64, 67, 68, 70, 74, 75, 76, 79, 80, 83, 84, 86, 90, 91, 92, 95, 96, 99, 100, 102, 106, 107, 108, 111, 112, 115, 116, 118, 122, 123, 124, 127, 128, 131, 132, 134, 138, 139, 140, 143, 144, 147, 148, 150, 154, 155, 156, 159, 160, 163, 164, 166, 170, 171, 172, 175, 176, 179, 180, 182, 186, 187, 188, 191, 192, 195, 196, 198, 202, 203, 204, 207, 208, 211, 212, 214, 218, 219, 220, 223, 224, 227, 228, 230, 234, 235, 236, 239, 240, 243, 244, 246, 250, 251, 252, 255, 256, 259, 260, 262, 266, 267, 268, 271, 272, 275, 276, 278, 282, 283, 284, 287, 288, 291, 292, 294, 298, 299, 300, 303, 304, 307, 308, 310, 314, 315, 316, 319, 320, 323, 324, 326, 330, 331, 332, 335, 336, 339, 340, 342, 346, 347, 348, 351, 352, 355, 356, 358, 362, 363, 364, 367, 368, 371, 372, 374, 378, 379, 380, 384]
        quantize_operators: ['!_add']
    ```
- bge-reranker-v2-m3模型，int8量化精度掉点验证，可使用以下量化参数实现混合精度量化，跳过一些层保留fp16
    ```yaml
    quantize:
        calibrate_mode: max
        quantize_per_channel: false
        overflow_adaptive: 1
        weight_scale: max
        calibrate_chunk_by: -1
        exclude_layers: [1, 2, 5, 6, 10, 11, 12, 15, 16, 19, 20, 22, 26, 27, 28, 31, 32, 35, 36, 38, 42, 43, 44, 47, 48, 51, 52, 54, 58, 59, 60, 63, 64, 67, 68, 70, 74, 75, 76, 79, 80, 83, 84, 86, 90, 91, 92, 95, 96, 99, 100, 102, 106, 107, 108, 111, 112, 115, 116, 118, 122, 123, 124, 127, 128, 131, 132, 134, 138, 139, 140, 143, 144, 147, 148, 150, 154, 155, 156, 159, 160, 163, 164, 166, 170, 171, 172, 175, 176, 179, 180, 182, 186, 187, 188, 191, 192, 195, 196, 198, 202, 203, 204, 207, 208, 211, 212, 214, 218, 219, 220, 223, 224, 227, 228, 230, 234, 235, 236, 239, 240, 243, 244, 246, 250, 251, 252, 255, 256, 259, 260, 262, 266, 267, 268, 271, 272, 275, 276, 278, 282, 283, 284, 287, 288, 291, 292, 294, 298, 299, 300, 303, 304, 307, 308, 310, 314, 315, 316, 319, 320, 323, 324, 326, 330, 331, 332, 335, 336, 339, 340, 342, 346, 347, 348, 351, 352, 355, 356, 358, 362, 363, 364, 367, 368, 371, 372, 374, 378, 379, 380, 384]
        quantize_operators: ['!_add']
    ```