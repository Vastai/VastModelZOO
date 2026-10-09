# VAMP

VAMP（VastAI Model Performance）是瀚博半导体VastAI-AIS团队提供的模型推理与性能精度测试工具，基于VastStreamX（vsx）流式推理框架构建，用于在瀚博GPU加速卡上对Build_In后端编译产出的模型进行推理、性能测试和精度测试。

VAMP集成于VVI（Vastai Versatile Inference）软件包的**vastpipe**组件中，随vastpipe组件一同安装发布。VAMP运行时依赖VastStreamX（vsx）和VastStream等VVI核心组件，需先激活vsx环境后方可使用。

- VAMP版本：`2.5.2_03603c8b`（2026-01-08）
- 所属VastPipe版本：`2.7.3`
- 源码仓库：[model_profiler](http://gitlabdev.vastai.com/AIS/VVI-SV100/model_profiler/-/blob/master/README.md)

## 特性

- 对所有支持的网络进行性能分析和功耗评估，可获得以下数据：
    - 指定batch size在单个或多个设备下的最大吞吐值，通过分配多个实例到不同die上可获得跨卡调度的最大性能
    - 相应吞吐下的端到端时延和模型推理时延统计，默认统计平均、最大、最小时延，同时统计 `50%`、`90%`、`95%`、`99%` 分位值
    - AI利用率，运行时采样硬件资源和状态统计
    - 芯片显存占用情况
    - 卡温度，监测运行状态下的平均温度
    - 卡功耗，监测运行状态下的平均功耗（注意单位为卡，一张卡可能有多个die）
- 性能评估的同时兼具精度评估功能：
    - 通过 `--datalist` 指定输入数据列表，支持 `numpy` 的 `npz` 格式
    - 通过 `--cache_input` 预加载数据到卡上（需注意数据量不超过显存大小）
    - 通过 `--cache_input_host` 将数据加载到host内存（适用于大数据集场景）
    - 通过 `--path_output` 指定结果输出路径
    - 通过 `--cache_output` 缓存输出数据到内存
- 多进程支持：通过 `-p, --processes` 参数指定进程数
    - 指定多个设备时，进程数限制为 ≤ 设备数，设备会被尽可能平均分配到不同进程
    - 指定单个设备时，进程数限制为 ≤ 实例数，不同实例会被尽可能平均分配到不同进程

## 版本说明

| 项目 | 说明 |
| :--- | :--- |
| 适配VVI版本 | VVI-26.02 |
| 不兼容版本 | VVI-26.08 |
| 下载地址 | [瀚博开发者中心 - VVI-26.02](https://developer.vastaitech.com/downloads/vvi?version_uid=535409016185163776) |

> **注意：** VAMP当前版本基于VVI-26.02发布，依赖该版本中的vsx、VastStream等组件，**不兼容VVI-26.08**。使用VVI-26.08环境运行VAMP可能导致功能异常或无法运行，请确保使用VVI-26.02版本。

## 工具安装

VAMP集成于vastpipe组件中，不单独发布。获取VVI-26.02部署软件包后，安装vastpipe组件即可使用VAMP。

- 获取VVI-26.02部署软件包，参考：[瀚博开发者中心](https://developer.vastaitech.com/downloads/vvi?version_uid=535409016185163776)
- 安装VastStream
    ```bash
    sudo ./ai-xxx.bin
    ```
- 安装VastStreamX
    - Python：`pip install vaststreamx-xxx.whl`
    - C++：`sudo ./vaststreamx-xxx.bin`
- 安装vastpipe（包含VAMP工具）
    ```bash
    sudo ./vastpipe-xxx.bin
    ```
- 解压VastPipe安装包并拷贝VAMP工具至vastpipe安装目录
    ```bash
    tar -xzf VastPipe_2.7.3.tar.gz
    sudo mkdir -p /opt/vastai/vastpipe/vastpipe/bin
    sudo cp VastPipe_2.7.3/vamp /opt/vastai/vastpipe/vastpipe/bin/
    ```

> 其中，xxx表示版本相关信息，请根据VVI-26.02软件包实际情况替换。


## 环境配置

VAMP运行前需激活VastStreamX环境，加载vsx动态库及vastpipe组件依赖：

```bash
source /opt/vastai/vaststreamx/vaststreamx/bin/activate.sh
source /opt/vastai/vastpipe/vastpipe/bin/activate.sh
```

激活后，以下环境变量将被设置：

| 环境变量 | 说明 |
| :--- | :--- |
| `VASTSTREAMX_ROOT` | VastStreamX安装根目录 |
| `VASTPIPE_ROOT` | vastpipe组件安装根目录 |
| `VASTPIPE_VERSION_FILE` | vastpipe版本文件路径 |
| `VASTSTREAMX_VERSION_FILE` | VastStreamX版本文件路径 |

> VAMP运行时依赖的动态库搜索路径包括：
> `/opt/vastai/vastpipe/vastpipe/calculators`、`/opt/vastai/vastpipe/vastpipe/lib`、`/opt/vastai/vaststream/lib`、`/opt/vastai/vaststreamx/vaststreamx/lib`

## 工具使用

```bash
# 命令格式
vamp -m <model_prefix> --vdsp_params <vdsp_params.json> [options]
```

### 参数说明

```bash
vamp --help
usage: vamp [options] ...
options:
  -m, --model_prefix        model prefix of the model suite files
      --model_json           json file of the model suite
      --model_param          parameter file of the model suite
      --model_lib            library file of the model suite
      --hwconfig             hw-config file of the model suite
      --vdsp_params          vdsp preprocess parameter file
  -b, --batch_size           profiling batch size of the model (default: 1)
      --force_batch_size     forced to use the exact batch size (default: 0)
  -i, --instance             instance number or range for each device, e.g. 1:4 (default: 1)
  -d, --device_id            device id(s) to run, e.g. 0 or 0:3 or [0,2] (default: 0)
  -c, --config               config file
  -g, --graph                [advanced] graph file to profile
  -f, --file_report          report file to save
  -s, --shape                model input shape list, e.g. [[1,256,256],[1,256,256]]
      --dtype                model input datatype (default: uint8)
      --iterations           iterations count for one profiling (default: 1024)
  -p, --processes            number of processes to run (default: 1)
      --verbose              verbose print (default: 0)
      --percentiles          percentiles of latency [50,90,95,99]
      --datalist             input data file list
      --cache_input          cache input data into device memory (default: 0)
      --cache_input_host     cache input data into host memory (default: 0)
      --cache_output         cache output data into memory (default: 0)
      --path_output          save result when datalist is specified
      --backend              inference backend, options [vsx=default, vastpipe] (default: vsx)
      --producer_mode        0: only one producer for all die; 1: every die has own producer (default: 0)
      --none_vaml            true: do not call vaml API; false: call vaml API (default: 0)
      --freqlist             set [OCLK, ODSPCLK, CCLK, DLC, SOC], e.g. [835,835,1000,750,830]
  -v, --version              get version information
  -t, --forward_time         forward time, eg. -t 15, forward time 15 seconds (default: 0)
      --forward_mode         forward mode 0: async, 1: sync (default: 0)
  -?, --help                 print this message
```

<details><summary><b>参数详细说明</b></summary>

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `-m, --model_prefix` | 模型前缀路径，指向VAMC编译产出的模型三件套（model_json、model_param、model_lib）所在目录 | 必填 |
| `--model_json` | 模型套件的JSON文件 | - |
| `--model_param` | 模型套件的参数文件 | - |
| `--model_lib` | 模型套件的库文件 | - |
| `--hwconfig` | 模型套件的硬件配置文件 | - |
| `--vdsp_params` | VDSP预处理算子参数配置文件（JSON格式） | 必填 |
| `-b, --batch_size` | 性能测试的批处理大小 | `1` |
| `--force_batch_size` | 强制使用指定的精确batch size | `0` |
| `-i, --instance` | 每个设备上的推理实例数，支持范围，如 `1:4` | `1` |
| `-d, --device_id` | 指定运行的设备ID，支持范围或列表，如 `0`、`0:3`、`[0,2]` | `0` |
| `-c, --config` | 配置文件路径，通过YAML配置文件指定参数 | - |
| `-g, --graph` | [高级] 指定Vastpipe的 `.pbtxt` graph文件进行profile | - |
| `-f, --file_report` | 测试报告输出文件路径（YAML格式） | - |
| `-s, --shape` | 模型输入shape列表，如 `[3,224,224]` 或多输入 `[[1,256,256],[1,256,256]]` | - |
| `--dtype` | 模型输入数据类型 | `uint8` |
| `--iterations` | 单次性能测试的推理迭代次数 | `1024` |
| `-p, --processes` | 并发进程数，多设备时限制≤设备数，单设备时限制≤实例数 | `1` |
| `--verbose` | 详细打印模式，开启时输出每个进程的统计信息 | `0` |
| `--percentiles` | 延迟百分位数，如 `[50,90,95,99]` | `[50,90,95,99]` |
| `--datalist` | 输入数据文件列表，用于精度测试，支持 `npz` 格式 | - |
| `--cache_input` | 将输入数据预加载到设备内存 | `0` |
| `--cache_input_host` | 将输入数据加载到主机内存（适用于大数据集） | `0` |
| `--cache_output` | 将输出数据缓存到内存 | `0` |
| `--path_output` | 推理结果输出目录，指定datalist时生效 | - |
| `--backend` | 推理后端，可选 `vsx`（默认）或 `vastpipe` | `vsx` |
| `--producer_mode` | 生产者模式，`0`：所有die共用一个生产者，`1`：每个die独立生产者 | `0` |
| `--none_vaml` | `true`：不调用vaml API，`false`：调用vaml API | `0` |
| `--freqlist` | 设置加速卡频率参数 `[OCLK, ODSPCLK, CCLK, DLC, SOC]`，如 `[835,835,1000,750,830]` | - |
| `-v, --version` | 打印版本信息 | - |
| `-t, --forward_time` | 持续推理时间（秒），如 `-t 15` 表示持续推理15秒 | `0` |
| `--forward_mode` | 推理模式，`0`：异步（默认），`1`：同步 | `0` |
| `-?, --help` | 打印帮助信息 | - |

</details>

### 性能测试输出指标

| 指标 | 说明 |
| :--- | :--- |
| throughput (qps) | 吞吐量 |
| e2e latency (us) | 端到端延迟（avg/min/max + 百分位） |
| model latency (us) | 模型推理延迟（avg/min/max） |
| ai utilize (%) | AI利用率 |
| die memory used (MB) | die显存使用量 |
| card power (W) | 卡功耗 |
| temperature (°C) | 卡温度 |

## 使用示例

### 单die性能测试

```bash
vamp -m /pathto/resnet50-int8-percentile-1_3_224_224-vacc/resnet50 \
    --vdsp_params /pathto/configs/vdsp_resnet50.json \
    -p 1 -b 8 -i 3 -d 0 --iterations 10240
```

输出示例：
```
Dump model info: /pathto/resnet50-int8-percentile-1_3_224_224-vacc/resnet50
[max_batch_size_]: 27
[input_count]: 1
input[0]: [3,224,224], dtype:u1
[output_count]: 1
output[0]: [1,1000], dtype:f2

*** number of instances in each device: 3 ****
  devices: [0]
  batch size: 8
  samples: 10240
  forwad time (s): 3.30204
  throughput (qps): 3101.12
  ai utilize (%): 98.0883
  card power (W): 43.7734
  temperature (°C): 47.7225
  die memory used (MB): 1212.67
  e2e latency (us):
    avg latency: 128843
    min latency: 9512
    max latency: 201022
  model latency (us):
    avg latency: 315
    min latency: 315
```

### 多die性能测试

```bash
vamp -m /pathto/resnet50-int8-percentile-1_3_224_224-vacc/resnet50 \
    --vdsp_params /pathto/configs/vdsp_resnet50.json \
    -p 1 -b 8 -i 3 -d 0:1 --iterations 10240
```

### 多die多进程性能测试

```bash
vamp -m /pathto/resnet50-int8-percentile-1_3_224_224-vacc/resnet50 \
    --vdsp_params /pathto/configs/vdsp_resnet50.json \
    -p 2 -b 8 -i 3 -d 0:1 --iterations 10240
```

### 性能（吞吐和时延） & 精度同步评估

> 吞吐模式：`-b` 设置较大batch size，`-i` 设置多个实例
> 时延模式：`-b 1 -i 1 --forward_mode 1`

```bash
# 吞吐模式
vamp -m /pathto/resnet50-int8-kl_divergence-3_32_32-vacc/resnet50 \
    -s [3,32,32] --vdsp_params ./data/vdsp_params/resnet_32.json \
    -b 16 -i 2 --datalist ./data/lists/cifar_inputs.txt \
    --path_output ./outputs/cifar --cache_input 1 -d [0,1] -p 2

# 时延模式
vamp -m /pathto/resnet50-int8-kl_divergence-3_32_32-vacc/resnet50 \
    -s [3,32,32] --vdsp_params ./data/vdsp_params/resnet_32.json \
    -b 1 -i 1 --datalist ./data/lists/cifar_inputs.txt \
    --path_output ./outputs/cifar --cache_input 1 -d [0,1] -p 1 --forward_mode 1
```

### 使用config文件

通过YAML配置文件指定参数，替代命令行参数：

```bash
vamp --config ./data/configs/resnet.yaml
```

## VastModelZOO中的应用

在VastModelZOO的CV模型部署中，VAMP用于Build_In后端模型的性能测试和精度测试。

### 性能测试

```bash
vamp -m deploy_weights/official_resnest_fp16/mod \
    --vdsp_params ../build_in/vdsp_params/official-resnest101-vdsp_params.json \
    -i 1 -p 1 -b 2 -s [3,224,224]
```

### 精度测试

> **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度。

1. 数据准备，生成推理数据`npz`以及对应的`datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/ILSVRC2012_img_val --target_path input_npz --text_path imagenet_npz.txt
    ```

2. vamp推理获取npz文件
    ```bash
    vamp -m deploy_weights/official_resnest_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-resnest101-vdsp_params.json \
        -i 8 -p 1 -b 22 -s [3,224,224] \
        --datalist imagenet_npz.txt --path_output output
    ```

3. 解析输出结果用于精度评估
    ```bash
    python ../../common/eval/vamp_npz_decode.py imagenet_npz.txt output imagenet_result.txt imagenet.txt
    ```

4. 精度评估
    ```bash
    python ../../common/eval/eval_topk.py imagenet_result.txt
    ```
