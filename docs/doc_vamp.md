# vamp

## 1. 资源获取

- 下载[vamp](http://192.168.20.17:3456/data/model_profiler/)内需要的相关版本
- 下载依赖[vastpipe 1.4.8](http://192.168.20.17:3456/vastpipe/1.4.8/vastpipe-1.4.8-b8caeeb2-Ubuntu-18.04-x86_64.tar.gz)

## 2. 安装使用
- 按以下步骤安装`vamp`
    ```shell
    # Step1 安装vastpipe

    # 创建文件夹，赋权限
    sudo mkdir -p /opt/vastai
    sudo chgrp -R sudo /opt/vastai
    sudo chmod -R 777 /opt/vastai
    
    # 解压vastpipe
    tar xzvf vastpipe-x.x.x-{commit}-Ubuntu-18.04-x86_64.tar.gz
    cd vastpipe-x.x.x-{commit}-Ubuntu-18.04-x86_64
    sudo chmod 777 install.sh
    sudo ./install.sh
    
    # 添加环境变量
    sudo chmod 777 /opt/vastai/vastpipe/vastpipe/bin/activate.sh

    sudo vim ~/.bashrc
    # 在~/.bashrc文件末尾添加以下两行环境变量
    export VASTPIPE_HOME=/opt/vastai/vastpipe/vastpipe
    source $VASTPIPE_HOME/bin/activate.sh
    
    # :wq保存后，source更新~/.bashrc文件
    source ~/.bashrc
    ```

    ```shell
    # Step2 安装vamp

    # 切换至vamp目录，将vamp移动到/opt/vastai/vastpipe/vastpipe/bin，使得vamp命令可在任何目录内使用
    cd vamp/x.x.x/
    sudo mv vamp /opt/vastai/vastpipe/vastpipe/bin
    sudo chmod a+x /opt/vastai/vastpipe/vastpipe/bin/vamp
    ```

- 参考vamp使用文档`x.x.x/README.md`，进行性能和精度测试
    ```shell
    # 性能
    ./vamp -m deploy_weights/resnet50-int8-percentile-3_224_224-vacc/resnet50 --vdsp_params vacc_code/vamp_info/vdsp_params.json -i 2 p 2 -b 1 -s [3,224,224]

    # 精度，需指定npz文件路径
    ./vamp -m deploy_weights/resnet50-int8-percentile-3_224_224-vacc/resnet50 --vdsp_params vacc_code/vamp_info/vdsp_params.json -i 2 p 2 -b 1 -s [3,224,224] --datalist ./data/lists/npz_datalist.txt --path_output ./save/resnet50
    ```
    >
    > 参数`i`，`p`，`b`的选择参考VAMP文档，以刚好使得板卡AI利用率达满为佳
    >

    其中，`--vdsp_params`为C++版的VDSP算子参数文件；`--datalist`为评估数据集原始图像转为npz文件后组成的文件路径，转换脚本可参考[image2npz.py](../classification/common/utils/image2npz.py)，形式如下：
    ```
    # npz_datalist.txt
    /home/simplew/code/model_check/SR/sr/pytorch-vdsr/Set5_scale4_npz/head_GT_scale_4.bmp.npz
    /home/simplew/code/model_check/SR/sr/pytorch-vdsr/Set5_scale4_npz/butterfly_GT_scale_4.bmp.npz
    /home/simplew/code/model_check/SR/sr/pytorch-vdsr/Set5_scale4_npz/woman_GT_scale_4.bmp.npz
    /home/simplew/code/model_check/SR/sr/pytorch-vdsr/Set5_scale4_npz/baby_GT_scale_4.bmp.npz
    /home/simplew/code/model_check/SR/sr/pytorch-vdsr/Set5_scale4_npz/bird_GT_scale_4.bmp.npz
    ```
    vmap指定验证数据集后，将保存结果至`--path_output`目录，每个图片的结果对应一个npz。解析npz结果与GT进行对比，即可获得评估指标，解析可参考[vamp_npz_decode.py](../classification/common/eval/vamp_npz_decode.py)脚本。
## 3. Tips
- 瀚博模型性能诊断工具，内部名称为`Model_Profiler`，对外名称为`VAMP`
