# vacc requirements
> just for ubuntu18.04

## compiler & sdk1.x

1. 获取发布包，[release_package](http://192.168.20.74/SW_DEV_AI_smoke_daily_export/)
2. 宿主机依赖
   1. cmake 3.10.2

    ```bash
    cd /home/vastai/cmake-3.10.2
    ./bootstrap
    make -j20
    sudo make install
    ```

   2. llvm 9.0.1 (optional)

    ```bash
    sudo apt-get install libllvm-9-ocaml-dev libllvm9 llvm-9 llvm-9-dev llvm-9-doc llvm-9-examples llvm-9-runtime
    ```

3. python

    ```bash
    pip install --no-cache-dir -r http://10.23.4.220:9011/vamc/tools/requirements.txt
    ```

4. set env

    ```bash
    export TVM_HOME="/home/lance/workspace/sw_release/1.3.0.RC2/tvm"
    export VASTSTREAM_PIPELINE=true
    export VASTSTREAM_HOME="/home/lance/workspace/sw_release/1.3.0.RC2/vastream/release/vacl"
    export VACC_IRTEXT_ENABLE=1
    export LD_LIBRARY_PATH=$TVM_HOME/lib:$VASTSTREAM_HOME/lib
    export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/vacc/python:$TVM_HOME/topi/python:${PYTHONPATH}:$VASTSTREAM_HOME/python
    ```

5. import test

    ```bash
    import torch
    import vacc
    vacc.__file__
    import vaststream
    vaststream.__file__
    ```

## compiler & sdk2.x

1. 获取发布包，[release_package](http://192.168.20.74/SW_DEV_AI_smoke_daily_export/)
2. 宿主机依赖
   1. cmake 3.10.2

    ```bash
    cd /home/vastai/cmake-3.10.2
    ./bootstrap
    make -j20
    sudo make install
    ```

   2. llvm 9.0.1 (optional)

    ```bash
    sudo apt-get install libllvm-9-ocaml-dev libllvm9 llvm-9 llvm-9-dev llvm-9-doc llvm-9-examples llvm-9-runtime
    ```

3. 安装

    ```bash
    sudo dpkg -i vaststream-sdk_xx-xx-xx-xx-xx_amd64.deb
    ```

4. python

    ```bash
    pip install --no-cache-dir -r http://10.23.4.220:9011/vamc/tools/requirements.txt
    pip install vaststream
    ```

5. set env

    ```bash
    export LD_LIBRARY_PATH=/home/lance/workspace/sw_release/1.3.0.RC2/tvm/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/home/lance/workspace/sw_release/1.3.0.RC2/tvm/python:/home/lance/workspace/sw_release/1.3.0.RC2/tvm/vacc/python:${PYTHONPATH}
    export LD_LIBRARY_PATH=/opt/vastai/vaststream2.0/lib:$LD_LIBRARY_PATH
    ```

6. import test

    ```bash
    import torch
    import vacc
    vacc.__file__
    import vaststream
    vaststream.__file__
    ```

## PCIE

1. 查看是否含有瀚博推理卡

   ```bash
    lspci -d:0100
   ```

2. 宿主机依赖

    ```bash
    sudo apt-get install dh-make dkms dpkg dpkg-dev python2 python3
    ```
3. 安装

    ```bash
    sudo dpkg -i vastai-pci_dkms_xx.xx.xx.xx_xx.deb
    ```

4. 查询安装

    ```bash
    dpkg --status vastai-pci-dkms

    #output
    Package: vastai-pci-dkms
    Status: install ok installed
    ……
    Version: xx.xx.xx.xx
    Provides: vastai-pci-modules (= xx.xx.xx.xx)
    Depends: dkms (>= 1.95)
    Description: vastai-pci driver in DKMS format.
    ```

5. 确认是否加载到内核

    ```bash
    lsmod | grep vastai_pci
    ```

6. 卸载驱动

    ```bash
    #查询安装包名字
    dpkg -l | grep vastai
    #卸载安装包
    sudo dpkg -r vastai-pci-dkms
    ```

7. 重启卡

    ```bash
    sudo chmod 666 /dev/kchar:0 && sudo echo reboot > /dev/kchar:0
    ```

8. docker

    ```bash
    sudo chmod +66 /dev/vacc0
    sudo chmod +66 /dev/vacc1
    ```

## tools
`\\192.168.29.20\Product\v1.3\ubuntu\vatools`


## reference

- [sdk doc](http://docs.vastai.com/V2_0/sdk/index_sdk_install_py.html)