
# qwen3_reranker 使用说明


## qwen3_reranker.py 命令参数说明

```bash
python qwen3_reranker.py -h

options:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        Model name
```


## qwen3_reranker.py 运行示例

```bash
sudo docker run --rm -it --net=host --privileged=true -v `pwd`:/workspace -v /logs:/logs harbor.vastaitech.com/ai_deliver/vllm_base:v0.9.2_cpu python /workspace/qwen3_reranker.py

```

## qwen3_reranker.py 运行结果示例
```bash
------------------------------
[0.9994966983795166, 0.99935382604599]
------------------------------
```