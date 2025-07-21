# 前置依赖说明
## 支持的加速卡
VA16: 支持DeepSeek系列(TP32), Qwen3(TP2), Qwen3(TP4)

VA1L: 支持Qwen3(TP2)

## 支持的 CPU 型号
- **Intel**: 
  - Xeon Platinum 8358
  - Xeon Gold 6330 
  - Xeon Gold 6430
  - Xeon Gold 6530
- **Hygon**: 
  - C86 7375 32-core @3.0GHz
  - C86-4G (OPN:7470) 48-core @2.6GHz (dual CPU)
- **AMD**: 
  - EPYC 7543 32-Core
  - EPYC 7352 24-Core
- **Phytium**: 
  - S5000C/64 @2.1GHz (dual CPU) * 2 

  注意：
  1. 绑核 taskset -c 0-63；
  2. 飞腾S5000C CPU die和die之间无法支持P2P， 而多张卡跑大模型，卡之间需要P2P通信，
  所以如果要在飞腾服务器上接V16，需要把switch 连接为级联模式（cpu - switch - swich），
  通过switch 支持P2P； 级联配置8张卡在同一个die下，8卡需要的MMIO 会大于1T， 
  而飞腾CPU 默认的配置MMIO 只有1TB， 会出现MMIO 空间不足的问题，
  需要飞腾修改PBF和BIOS提升单die MMIO 空间到4TB 解决。

## 支持的 OS 版本

| OS Version            | Kernel Version                                  |
|-----------------------|------------------------------------------------|
| Ubuntu 22.04          | `5.15.0-119-generic`, `5.15.0-139-generic`     |
| UOS Server 20         | `4.19.90-2403.3.0.0270.87.uel20.x86_64`        |
| KeyarchOS-5.8-SP2-U1  | `5.10.134-17.2.2.kos5.x86_64`                  |
| Kylin V10             | `4.19.90-89.11.v2401.ky10.aarch64`             |

## 一键安装前置依赖
注意：  一键安装脚本仅针对一体机配置（8卡VA16机器）

### 1. Ubuntu 22.04 
- 一键安装目前仅支持x86 架构下 Ubuntu 22.04 系统

### 2. System Configuration
- Disable IOMMU

### 3. Containerization
- **Docker**:
  - Packages: `docker-ce`, `docker-ce-cli`, `containerd.io`
  - Minimum version: 26.1.3

- **Docker Compose**:
  - Minimum version: 1.29.2
  - Installation:
    ```bash
    wget https://github.com/docker/compose/releases/download/v2.26.1/docker-compose-linux-x86_64 -O /usr/bin/docker-compose
    chmod +x /usr/bin/docker-compose
    ```

### 4. System Tools
```bash
apt-get install -y \
    linux-tools-common \
    linux-tools-$(uname -r) \
    g++ \
    gzip \
    tar \
    net-tools \
    libssl-dev \
    make 
```

### 5. Python Environment
- **Python 3.10**
- **pip3**

### 6. Python Packages
```bash
pip install openai==1.93.0 -i https://mirrors.ustc.edu.cn/pypi/web/simple
```

### 7. Utility Tools
- md5sum
- base64

