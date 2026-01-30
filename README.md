<div id=top align="center">

![logo](./images/index/logo.png)
[![License](https://img.shields.io/badge/license-Apache_2.0-yellow)](LICENSE)
[![company](https://img.shields.io/badge/company-vastaitech.com-blue)](https://www.vastaitech.com/)
[![deepwiki](https://img.shields.io/badge/deepwiki-VastModelZOO-white)](https://deepwiki.com/Vastai/VastModelZOO/)
[![pages](https://img.shields.io/badge/model_list-vastai.github.io/VastModelZOO-pink)](https://vastai.github.io/VastModelZOO/)

</div>

---

`VastModelZOO`是`瀚博半导体VastAI`维护的AI模型平台，提供了人工智能多个领域（CV、AUDIO、NLP、LLM、MLLM等）的开源模型在瀚博GPU芯片上的部署、训练示例。

`VastModelZOO`旨在基于`瀚博半导体VastAI`的硬件产品和软件SDK，展示最佳的编程实践，以达成模型的快速移植和最优性能。

为方便大家使用`VastModelZOO`，我们将持续增加典型模型和基础插件。


## 依赖软件

- 基于`瀚博半导体VastAI`的硬件产品使用`VastModelZOO`前，需联系销售代表获取`瀚博开发者中心`版本权限

- 访问[瀚博开发者中心](https://developer.vastaitech.com/downloads/vvi?version_uid=)，获取`VVI(Vastai Versatilve Inference)`部署软件包


## 快速安装

获取部署软件包后安装流程如下。

<details><summary><b>步骤 1.</b> 安装驱动</summary>

1. 查询是否安装加速卡

    ```shell
    lspci -d:0100 |wc -l
    ```

2. 查询是否安装驱动

    ```shell
    lsmod | grep -i vastai_pci
    ```

3. 查询驱动版本

    ```shell
    cat /dev/vastai0_version | grep "Driver"
    ```

4. 安装驱动

- 部署LLM/VLM模型

    ```shell
    sudo ./vastai_driver_install_xxx.run install --setkoparam "dpm=1"
    ```

- 部署非LLM/VLM模型

    ```shell
    sudo ./vastai_driver_install_xxx.run install
    ```

</details>

<details><summary><b>步骤 2.</b> 设置加速卡参数</summary>

1. 查询加速卡信息

    ```shell
    sudo vasmi list
    ```

2. (可选) 开启 DPM

    > 仅针对LLM/VLM模型需要开启 DPM

    ```shell
    sudo vasmi setconﬁg dpm=enable -d all
    ```

3. 根据业务情况设置加速卡Bbox模式

    ```shell
    sudo vasmi setcardmode <Card Mode> -d <Device ID> -y
    ```

    > Card Mode可根据 `sudo vasmi setcardmode --help` 查询获取

4. 使能日志记录等监控功能

    ```shell
    nohup sudo valogger &
    ```

</details>

<details><summary><b>步骤 3.</b> 部署模型运行环境（ARM/X86）</summary>

- Build_In 后端模型运行环境部署

  1. 安装 VastStream

        ```shell
        sudo ./ai-xxx.bin
        ```

  2. 安装 VAMC
        ```shell
        pip install vamc-xxx.whl
        ```

  3. 安装 VastStreamX

     - Python：`pip install vaststreamx-xxx.whl`
     - C++：`sudo ./vaststreamx-xxx.bin`

  4. 安装 VastGenX（仅LLM/VLM）
        ```shell
        pip install vastgenx-xxx.whl
        ```

  5. 安装 VastGenServer（仅Text2vec）
        ```shell
        pip install vastgenserver-xxx.whl
        ```

- vLLM 后端模型运行环境部署

  1. 安装 torch_vacc
        ```shell
        pip install torch_vacc-xxx.whl
        ```

  2. 安装 vLLM_vacc
        ```shell
        pip install vllm_vacc-xxx.whl
        ```

> 若 VLM 模型为 vLLM+Build_In 的混合部署方案，需安装 Build_In 后端模型运行环境部署中的1、2、3、4 和 vLLM 后端模型环境部署中的1、2

</details>

> 详细安装及使用说明可参考对应组件的文档。
> 其中，xxx表示版本相关信息，请根据实际情况替换。


## 模型列表

- 检索模型列表，请访问：[📚 vastai.github.io/VastModelZOO](https://vastai.github.io/VastModelZOO/)


## 免责声明
- `VastModelZOO`提供的模型仅供您用于非商业目的，请参考原始模型来源许可证进行使用
- `VastModelZOO`描述的数据集均为开源数据集，如您使用这些数据集，请参考原始数据集来源许可证进行使用
- 如您不希望您的数据集或模型公布在`VastModelZOO`上，请您提交issue，我们将尽快处理


## 使用许可
- `VastModelZOO`提供的模型，如原始模型有许可证描述，请以该许可证为准
- `VastModelZOO`遵循[Apache 2.0](LICENSE)许可证许可