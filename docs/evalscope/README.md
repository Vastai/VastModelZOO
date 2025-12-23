# EvalScope

EvalScope 是一个用于评估生成模型性能的工具，支持评估LLM和VLM模型。

## 工具安装
- 安装evalscope，参考：[installation](https://evalscope.readthedocs.io/zh-cn/latest/get_started/installation.html)


## 精度测试

- 启动api server服务,使用[vastgenx](../vastgenx/README.md)或[vLLM](https://github.com/vllm-project/vllm)
- 配置测评数据集及采样参数等信息，执行脚本获取精度测评结果
    > 数据集将自动下载
    - [precision_llm.py](./precision_llm.py)
    - [precision_vlm.py](./precision_vlm.py)
    - [precision_embedding.py](./precision_embedding.py)
    - [precision_reranker.py](./precision_reranker.py)
    ```bash
    cd docs/evalscope
    #测试llm模型
    python precision_llm.py
    #测试vlm模型
    python precision_vlm.py
    # 测试embedding模型
    python precision_embedding.py
    # 测试reranker模型
    python precision_reranker.py

    ```

## 性能测试
- 启动api server服务,使用[vastgenx](../vastgenx/README.md)或[vLLM](https://github.com/vllm-project/vllm)
- 配置测评数据集及采样参数等信息，执行脚本获取性能测评结果
    - [pref_llm.py](./pref_llm.py)
    - [pref_vlm.py](./pref_vlm.py)

    ```bash
    cd docs/evalscope
    #测试llm模型
    python pref_llm.py
    # 测试vlm模型
    python pref_vlm.py
    ```