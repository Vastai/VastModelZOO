
# qwen3_embedding 使用说明


## qwen3_embedding.py 命令参数说明

```bash
python qwen3_embedding.py -h

options:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        Model name
```


## qwen3_embedding.py 运行示例

```bash
sudo docker run --rm -it --net=host --privileged=true -v `pwd`:/workspace -v /logs:/logs harbor.vastaitech.com/ai_deliver/vllm_base:v0.9.2_cpu python /workspace/qwen3_embedding.py

```

## qwen3_embedding.py 运行结果示例
```bash
[[0.7631064653396606, 0.14042744040489197], [0.1342908889055252, 0.6023261547088623]]
```