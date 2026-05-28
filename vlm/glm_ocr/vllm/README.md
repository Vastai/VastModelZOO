# vLLM部署

## 模型支持

  |model | huggingface  | modelscope | parameter | dtype| arch |
  | :--- | :--- | :-- | :-- | :-- | :-- |
  |GLM-OCR | [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | [ZhipuAI/GLM-OCR](https://modelscope.cn/models/ZhipuAI/GLM-OCR) | 0.9B | BF16 |VLM-Dense-GQA |
  |GLM-OCR-FP8 | [vastai-ais/GLM-OCR-FP8](https://huggingface.co/vastai-ais/GLM-OCR-FP8) | [vastai-ais/GLM-OCR-FP8](https://modelscope.cn/models/vastai-ais/GLM-OCR-FP8) | 0.9B | FP8 |VLM-Dense-GQA |

## 使用限制

  | model | parallel | seq limit | mtp | tips|
  |:--- |:--- | :-- | :-- | :-- |
  | GLM-OCR* | tp1/2/4 | max-model-len 128k | ❌ | max-concurrency 4|

> - max-input-len: 最大输入长度
> - max-model-len: 最大上下文长度
> - mtp: Multi-Token Prediction，多token预测模式 (当前VACC暂未适配GLM-OCR的MTP特性)
> - max-concurrency: 最大并发
> - 对于超过上下文长度的请求，内部会拦截不做处理，需要客户端自行处理


## 模型下载
1. 通过hf-mirror下载

- 参考[hf-mirror](https://hf-mirror.com/)下载权重
  ```shell
  wget https://hf-mirror.com/hfd/hfd.sh
  chmod a+x hfd.sh
  export HF_ENDPOINT=https://hf-mirror.com
  apt install aria2
  ./hfd.sh zai-org/GLM-OCR -x 10 --local-dir GLM-OCR
  ```

2. 或通过modelscope下载

- 参考[modelscope](https://modelscope.cn/docs/models/download)下载权重

  ```shell
  pip install modelscope -i https://mirrors.ustc.edu.cn/pypi/web/simple
  export PATH=$PATH:~/.local/bin
  modelscope download --model ZhipuAI/GLM-OCR --local_dir GLM-OCR
  ```

## 模型量化
- 参考：[quant.md](./quant.md)

## 启动模型服务

1. 参考官方启动命令：[vllm](https://docs.vllm.ai/en/latest/cli/#bench)

  ```shell
    docker run -it --rm --shm-size=256g --ipc=host --network=host \
    --ulimit memlock=-1 --privileged --ulimit stack=67108864 \
    -v /FS03:/FS03 \
    --name glm_ocr harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-xx.xx bash

    VACC_VISIBLE_DEVICES=0,1,2,3 vllm serve ./weights/GLM-OCR/ \
    --served-model-name glm-ocr \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --trust-remote-code \
    --enforce_eager \
    --no-async-scheduling \
    --port 8000
  ```

> 参数说明参考vLLM官方。


## 模型性能测试

> 模型性能包含吞吐和推理时延，可通过 vLLM 服务加载模型，并使用 vLLM 自带框架进行性能测试。

1. 参考vLLM文档测试模型性能：[benchmarking/cli](https://docs.vllm.ai/en/latest/benchmarking/cli/)

    ```shell
    vllm bench serve \
    --model ./weights/GLM-OCR/ \
    --served-model-name glm-ocr \
    --base_url http://127.0.0.1:8000 \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --num-prompts 50 \
    --max-concurrency 1 \
    --dataset-name random-mm \
    --random-input-len 128 \
    --random-output-len 1024 \
    --random-range-ratio 0.0 \
    --num-warmups 8 \
    --random-mm-base-items-per-request 1 \
    --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
    --random-mm-bucket-config '{(1080, 1920, 1): 1.0}' \
    --request-rate inf \
    --ignore-eos \
    --seed 42
    ```

> 参数说明参考vLLM官方。

## 模型精度测试
- 测评工具：[OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- 测评数据集：[OmniDocBench v1.6](https://modelscope.cn/datasets/OpenDataLab/OmniDocBench)

    ```shell
    modelscope download --dataset OpenDataLab/OmniDocBench --local_dir OmniDocBench
    ```

- 推理工具
    > - GLM-OCR，使用vLLM后端，部署在VACC-VA16
    > - [PP-DocLayoutV3](https://modelscope.cn/models/PaddlePaddle/PP-DocLayoutV3_safetensors)，使用transformer后端，部署在CPU

    ```shell
    git clone https://github.com/zai-org/glm-ocr.git
    cd glm-ocr

    pip install uv -i https://mirrors.aliyun.com/pypi/simple/
    uv venv --python 3.12 --seed && source .venv/bin/activate
    uv pip install -e '.[all]' -i https://mirrors.aliyun.com/pypi/simple/
    ```

- 推理测试

    ```shell
    # glmocr parse examples/source \
    glmocr parse datasets/OmniDocBench/images \
    --set pipeline.maas.enabled false \
    --set pipeline.ocr_api.api_host 127.0.0.1 \
    --set pipeline.ocr_api.api_port 8000 \
    --set pipeline.ocr_api.model glm-ocr \
    --set pipeline.ocr_api.request_timeout 6000 \
    --set pipeline.result_format.enable_merge_formula_numbers false \
    --set pipeline.layout.model_dir ./PaddlePaddle/PP-DocLayoutV3_safetensors \
    --layout-device cpu \
    --output ./results_samples/
    ```

- 精度测评
    ```shell
    # 整理md结果至同一文件夹
    mkdir results_md
    cp -r ./results/*/*.md results_md
    ```

    ```shell
    # OmniDocBench工具中CDM指标测评依赖库安装比较复杂，建议使用docker方式测评，此步骤无需显卡
    # /cx8k/fs101/GLM-OCR/datasets/OmniDocBench/OmniDocBench.json：原始数据集标签
    # /cx8k/fs101/GLM-OCR/results_md：`copy md`步骤的md文件夹路径
    # /cx8k/fs101/GLM-OCR/results_metric：此工具生成文件保存路径

    sudo docker run -it \
    --entrypoint bash \
    -v /cx8k/fs101/GLM-OCR/datasets/OmniDocBench/OmniDocBench.json:/workspace/gt/your_gt.json:ro \
    -v /cx8k/fs101/GLM-OCR/results_md:/workspace/data_md/predictions:ro \
    -v /cx8k/fs101/GLM-OCR/results_metric:/workspace/result \
    docker.gh-proxy.org/ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204 \
    -c 'cat > configs/custom.yaml << "EOF"
    end2end_eval:
    metrics:
        text_block:
        metric: [Edit_dist]
        display_formula:
        metric: [Edit_dist, CDM]
        table:
        metric: [TEDS, Edit_dist]
        reading_order:
        metric: [Edit_dist]
    dataset:
        dataset_name: end2end_dataset
        ground_truth:
        data_path: ./gt/your_gt.json
        prediction:
        data_path: ./data_md/predictions
        match_method: quick_match
        match_workers: 4
        quick_match_truncated_timeout_sec: 300
        timeout_fallback_max_chunk_span: 10
        timeout_fallback_order_penalty: 0.10
    EOF
    python pdf_validation.py --config configs/custom.yaml'
    ```

- 指标统计

    > - 以上步骤将在`results_metric`文件夹内生成数个jsonl文件
    > - 通过此处脚本提取最终指标

    ```python

    import os
    import argparse
    import pandas as pd
    import numpy as np
    import json


    def main():
        parser = argparse.ArgumentParser(description="Generate result tables from OCR metric JSON files.")
        parser.add_argument("--result-folder", type=str, default="results_metric",
                            help="Folder containing the metric result JSON files.")
        parser.add_argument("--match-name", type=str, default="quick_match",
                            help="Match name used in result filenames.")
        parser.add_argument("--ocr-types", type=str, default='{"GLM-OCR": "predictions"}',
                            help='JSON string mapping OCR display names to result file prefixes.')
        args = parser.parse_args()

        ocr_types_dict = json.loads(args.ocr_types)
        result_folder = args.result_folder
        match_name = args.match_name

        dict_list = []
        for ocr_type in ocr_types_dict.values():
            result_path = os.path.join(result_folder, f'{ocr_type}_{match_name}_metric_result.json')

            with open(result_path, 'r') as f:
                result = json.load(f)

            save_dict = {}
            for category_type, metric in [("text_block", "Edit_dist"), ("display_formula", "CDM"), ("table", "TEDS"), ("table", "TEDS_structure_only"), ("reading_order", "Edit_dist")]:
                if metric == 'CDM' or metric == "TEDS" or metric == "TEDS_structure_only":
                    if result[category_type]["page"].get(metric):
                        save_dict[category_type+'_'+metric] = result[category_type]["page"][metric]["ALL"] * 100   # page级别的avg
                    else:
                        save_dict[category_type+'_'+metric] = '-'
                else:
                    save_dict[category_type+'_'+metric] = result[category_type]["all"][metric].get("ALL_page_avg", np.nan)

            dict_list.append(save_dict)

        df = pd.DataFrame(dict_list, index=ocr_types_dict.keys()).round(3)
        df['overall'] = ((1-df['text_block_Edit_dist'])*100 + df['display_formula_CDM'] + df['table_TEDS'])/3

        print(df)


    if __name__ == "__main__":
        main()

    # python generate_result_tables.py --result-folder results_metric
    ```

- 精度结果

> 端到端评测是对模型在PDF页面内容解析上的精度作出的评测。以模型输出的对整个PDF页面解析结果的Markdown作为Prediction。Overall指标的计算方式如下

```math
\text{Overall} = \frac{(1-\textit{Text Edit Distance}) \times 100 + \textit{Table TEDS} +\textit{Formula CDM}}{3}
```

| Model-Backend       | **Overall↑** | **Text****Edit****↓** | **Formula****CDM****↑** | **Table****TEDS****↑** | TableTEDS-S↑ | Read OrderEdit↓ | Model Infer Cost                         |
| ------------------- | ------------ | --------------------- | ----------------------- | ---------------------- | ------------ | --------------- | ---------------------------------------- |
| Official            | 95.22        | 0.044                 | 97.18                   | 92.83                  | 95.39        | 0.133           | MaaS API                                 |
| NVIDIA-H800-BF16-TP1 | 95.468       | 0.04                  | 96.923                  | 93.482                 | 96.009       | 0.141           | --layout-device cuda:1 12min             |
| NVIDIA-H800-BF16-TP1 | 95.391       | 0.039                 | 96.536                  | 93.537                 | 96.11        | 0.139           | --layout-device cpu 15.5min              |
| NVIDIA-H800-FP8-TP1  | 95.399       | 0.039                 | 96.725                  | 93.371                 | 95.884       | 0.139           | --layout-device cpu                      |
| VACC-VA16-BF16-TP1   | 95.647       | 0.038                 | 97.173                  | 93.57                  | 96.035       | 0.139           | --layout-device cpu 13h26min |
| VACC-VA16-BF16-TP2   | 95.611       | 0.039                 | 97.13                   | 93.602                 | 96.093       | 0.139           | --layout-device cpu 9h                   |
| VACC-VA16-BF16-TP4   | 95.635       | 0.039                 | 97.204                  | 93.6                   | 96.088       | 0.139           | --layout-device cpu 6h33min              |
