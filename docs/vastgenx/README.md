# vastgenx

VastAI-AIS团队开发LLM、VLM模型部署工具，在VastAI硬件上推理，提供兼容OpenAI API的接口，并提供WebUI服务。

## 工具安装
- 获取安装包
- pip安装
    ```bash
    pip install vastgenx-1.x.0+xx.xx-cp310-cp310-linux_x86_64.whl
    ```

## 工具使用

```bash
# 命令格式
vastgenx serve [--model MODEL] [--host HOST] [--port PORT] [--batch_size BATCH_SIZE] [--llm_devices LLM_DEVICES]   [--vit_model VIT_MODEL] [--vit_devices VIT_DEVICES] [--min_pixels MIN_PIXELS] [--max_pixels MAX_PIXELS]
```

其中
```bash
--model 指向vastai模型文件夹，若模型为vlm，则vit模型文件夹也可以放在该参数指定文件夹里
--host 服务ip地址，默认值：0.0.0.0
--port 服务端口号，默认值：9000
--batch_size 模型batch_size,默认值: 16
--llm_devices 指定 llm 模型运行在哪些die上，默认值:[0~tp]
--vit_model 指定vit模型所在文件夹，若不指定，则在--model参数文件夹里找vit模型。只有模型为vlm时，该参数生效
--vit_devices 指定 vit 模型运行在哪些die上，只有模型是vlm时生效，默认使用 llm_devices[0]
--min_pixels 指定 vit 模型 min_pixels 参数，只有模型是vlm时生效，默认从preprocess_config.json里读取
--max_pixels 指定 vit 模型 max_pixels 参数，只有模型是vlm时生效，默认从preprocess_config.json里读取
```

### api server

```bash
# llm
vastgenx serve --model ai300/Qwen2.5-7B-Instruct-int8-tp8-2048-4096 \
--port 9900 \
--llm_devices "[0,1,2,3,4,5,6,7]" 

# vlm
vastgenx serve --model ai300/qwen2_vl_7b_llm_28layer_2048_4096_tp4_int8 \
--vit_model ai300/qwen2_vl_7b_visual_32layer_5120_fp16 \
--port 9900 \
--llm_devices "[0,1,2,3]" \
--vit_devices "[4]" \
--min_pixels 78400 \
--max_pixels 921600
```


### webui
```bash 
# 命令格式
vastgenx webui [--api_base_url API_BASE_URL] [--host HOST] [--port PORT]
```

其中
```bash 
--api_base_url 指定vastgenx 模型服务地址与端口号
--host webui服务ip地址，默认值:0.0.0.0
--port webui服务端口号，默认值:9600
```

命令示例
```bash
vastgenx webui --api_base_url http://127.0.0.1:9900/ --port 9600
```

### 精度测试
#### 基于evalscope
- 安装evalscope，参考：[installation](https://evalscope.readthedocs.io/zh-cn/latest/get_started/installation.html)
- 参考前文启动api server服务
- 配置测评数据集及采样参数等信息，执行脚本获取精度测评结果
    > 数据集将自动下载
    - [precision_llm.py](./evalscope/precision_llm.py)
    - [precision_vlm.py](./evalscope/precision_vlm.py)
    ```bash
    python precision_llm.py
    python precision_vlm.py
    ```

### 性能测试
#### 基于evalscope
- 同`精度测试`，安装`evalscope`和启动api server服务
- 配置测评数据集及采样参数等信息，执行脚本获取性能测评结果
    - [pref_llm.py](./evalscope/pref_llm.py)
    - [pref_vlm.py](./evalscope/pref_vlm.py)

    ```bash
    cd docs/vastgenx/evalscope
    python pref_llm.py
    python pref_vlm.py
    ```
#### 基于脚本
- 启动api server服务
- 配置测评参数等信息，执行脚本获取性能测评结果
    - [benchmark_llm.sh](./script/benchmarks/benchmark_llm.sh)
    - [benchmark_vlm.sh](./script/benchmarks/benchmark_vlm.sh)

    ```bash
    cd docs/vastgenx/script/benchmarks
    ./benchmark_llm.sh
    ./benchmark_vlm.sh
    ```