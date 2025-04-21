# VastAI Software Stack

## Introduce

### VastStream
VastStream软件栈包括SDK开发套件和Tools工具链。
SDK开发套件包括视频处理库、图像处理库、算子库、模型量化工具、编译器、运行时系统及硬件驱动，对外提供不同层次抽象与封装的API，以方便开发者灵活使用。
Tools工具链提供了包括应用开发、功能调试、性能调优等在内的一系列工具。

### VAMC
VAMC（Vastai Model Converter）是瀚博推出的一款模型转换工具，可以将开源框架（如Pytorch、TensorFlow、ONNX等）的网络模型转换为瀚博硬件设备支持的统一运行时，在瀚博硬件设备上执行前向推理。

1. 获取大模型编译工具安装包，按文档安装
3. 量化
    - 配置模型量化参数：[vamc_quant.yaml](./vamc/vamc_quant.yaml)，配置量化类型，量化硬件、数据集等配置
    ```bash
    # 如需编译int8模型，需先量化
    vamc quant vamc/vamc_quant.yaml
    ```
4. 编译
    - 配置模型编译参数：[vamc_config.yaml](./vamc/vamc_config.yaml)，按实际场景配置参数表
    ```bash
    vamc compile vamc/vamc_config.yaml
    ```

### VastLLMDeploy

VastLLMDeploy是瀚博推出的一款大语言模型部署工具，集成了瀚博LLM系列模型推理、精度测试和性能测试功能。
VastLLMDeploy支持FastAPI 推理服务化，用户可通过FastAPI服务或本地加载LLM模型，测试LLM模型精度、性能；同时，支持WebUI服务部署，用户可通过WebUI访问FastAPI服务。

1. 获取大模型部署工具安装包，按文档安装
2. 配置模型推理参数：[llmdeploy_config.yaml](./llmdeploy/llmdeploy_config.yaml)，按实际场景配置参数表的`model，api，benchmark和eval字段`
3. 启动api服务
    ```bash
    llmdeploy api llmdeploy/llmdeploy_config.yaml
    ```
4. 推理测试
    - 使用webui界面测试服务
        ```bash
        # 待api服务启动后，再启动webui服务
        llmdeploy webui llmdeploy/llmdeploy_config.yaml
        ```
    - 使用api调用测试：[test_api.py](./llmdeploy/test_api.py)
        ```bash
        # 注意api端口号
        python llmdeploy/test_api.py
        ```
5. 性能测试
    ```bash
    llmdeploy benchmark llmdeploy/llmdeploy_config.yaml
    ```

6. 精度测试
    - 下载评估评估数据集：[OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)
    ```bash
    unzip OpenCompassData-core-20240207.zip
    # 指定yaml配置表内的eval.datasets_dir字段为解压后路径
    ```

    ```bash
    llmdeploy eval llmdeploy/llmdeploy_config.yaml
    ```
    > Tips
    > 
    > - 测试不同数据集，注意修改配置表的`eval.config_path`字段Python文件`./configs/eval/eval_demo.py`内代码
    > - 精度测试耗时较长，请耐心等待



### VastStreamX
VastStreamX是VastStream SDK的更高层次封装，提供了Stream管理、内存管理、模型加载与执行、算子加载与执行、多媒体数据处理等API。

### VAMP
VAMP（Vastai Model Profiler）是瀚博推出的一款性能分析和功耗测试工具（暂不支持大模型测试）
- 支持测试模型在不同Batch size以及不同Die下的性能，包括测试模型的最大吞吐量、推理时延以及端到端的时延。同时支持统计指定分位置的模型推理时延和端到端的时延
- 支持统计模型运行时瀚博加速卡的AI利用率、显存占用情况、温度以及功耗
- 支持在极致性能的情况下测试模型精度，同时支持预加载数据、指定输入数据列表等
- 支持多进程运行模型

1. 获取工具安装包，按文档安装
2. 性能测试
    ```bash
    vamp -m vacc_deploy/bge-m3-512-fp16/mod \
    --vdsp_params embedding-vdsp_params.json \
    -i 1 p 1 -b 1 -s [[1,512],[1,512],[1,512],[1,512],[1,512],[1,512]] --iterations 1024
    ```

### TORCH_VACC

Torch_VACC是瀚博在pytorch基础上开发的插件工具包，拓展pytorch以适应瀚博生态链硬件、算子。

### VLLM_VACC
VLLM_VACC是瀚博在开源vllm基础上开发的插件工具包，拓展vllm以适应瀚博生态链硬件、算子，实现大模型推理，依赖Torch_VACC。


## Download
- 请移步瀚博开发者中心，下载所需工具版本及文档：[devops.vastai.com](http://devops.vastai.com/artifact/project?page=3&limit=10)