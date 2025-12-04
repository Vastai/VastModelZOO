# official center_point deploy

## 特别提示

- 本模型以官方为原型：[OpenPCDet/tools/cfgs/waymo_models/centerpoint_pillar_1x.yaml](https://github.com/open-mmlab/OpenPCDet/blob/master/tools/cfgs/waymo_models/centerpoint_pillar_1x.yaml)，使用kitti公开数据训练的模型。

  - 由于OpenPCDet官方并未提供预训练模型，且本文档探究的是为了演示如何将官方模型部署到`瀚博VACC硬件`上，因此本模型是**未收敛**模型，模型精度非最佳。
  - 此处模型结构[kitti_centerpoint_pillar_1x_model.yaml](config/kitti_centerpoint_pillar_1x_model.yaml)，相较于官方原始模型，做了以下修改，以便在`瀚博VACC硬件`上部署。
    - 训练数据集修改为kitti数据集，并修改了数据集的配置文件
    - NUM_FILTERS: [ 64，64 ] -> [ 64 ]

- 运行本实例需要准备三个环境

  - `OpenPCDet环境`：用于精度验证、导出合并onnx模型和pytorch推理。需要cuda的环境，建议单独准备一个cuda环境，避免和vacc环境冲突。精度测试一般把vacc测试结果拷贝到pcdet环境中去操作的

  - `VACC环境`：用于模型转换和在`瀚博VACC硬件`推理，参考`部署软件整包`中的VAMC文档安装

  - `桌面环境`：主要通过`open3d`包来展示点云和检测结果。请使用`pip install open3d`来安装`open3d`模块，并注意代码中有torch的依赖，可以安装cpu版本的torch

  - 需要在`OpenPCDet环境`和 `VACC环境`都下载本工程源码。并都在center_point/workspace目录下载好测试数据集和模型

- 后续文档的操作，都是基于center_point/workspace目录为工作目录。建议提前建好workspace目录

## step.1 安装OpenPCDet环境

- 安装OpenPCDet环境，需有CUDA环境机器，具体参考官方安装步骤：[docs/INSTALL.md](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md)
  ```bash
  git clone https://github.com/open-mmlab/OpenPCDet.git
  cd OpenPCDet
  git checkout 8caccce

  conda create -n openpcdet python=3.10
  conda activate openpcdet
  # pip3 install torch torchvision
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

  pip install spconv-cu124
  python setup.py develop

  # https://github.com/open-mmlab/OpenPCDet/issues/1583
  pip install setuptools==58 av2 kornia==0.5.8 onnxsim
  pip install -r requirements.txt
  ```

## step.2 源码修改和onnx导出

> 注意此操作在`OpenPCDet环境`下进行

- 为适配VACC和导出onnx文件，需进行适当修改源码，详见：[modify_detail.md](./modify_detail.md)
  > - 注意所有带"#export onnx"的注释，都代表该部分代码修改仅仅是为了在导出onnx模型时使用
  > - 导出onnx模型后，如果需要对原始的pth模型进行推理，需要把注释去掉，并还原回原来的代码，否则pth模型推理会出现错误

| weight                                                                                                                   | tips                                  |
| :----------------------------------------------------------------------------------------------------------------------- | :------------------------------------ |
| [kitti_centerpoint_pillar_1x.pth](https://drive.google.com/drive/folders/1jODyyyaGN8LC7Hf-dwNOudKYYm_MPQP6?usp=sharing)  | 详见前述`特别提示`，训练获得torch权重 |
| [kitti_centerpoint_pillar_1x.onnx](https://drive.google.com/drive/folders/1jODyyyaGN8LC7Hf-dwNOudKYYm_MPQP6?usp=sharing) | 最终onnx                              |

## step.3 准备数据集

- `VACC环境`只做模型转换和推理，只需要下载**评估数据集**、**校正数据集**即可。
  - 只需要拷贝kitti/testing/velodyne数据到vacc环境，作为评估数据集即可
- `OpenPCDet环境`需要下载下面列出的所有资源。
- `桌面环境`只需要把需要展示的数据拷贝过去即可。

1. 获取KITTI数据集：[KITTI 3D object detection dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

- 所需数据如下：

  - 彩色图像数据（12GB）， data_object_image_2.zip
  - 点云数据（29GB），data_object_velodyne.zip
  - 相机矫正数据（16MB），data_object_calib.zip
  - 标签数据（5MB），data_object_label_2.zip

  > 其中彩色图像数据、点云数据、相机矫正数据均包含training（7481）和testing（7518）两个部分，标签数据只在training数据中。

  ```bash
  mv data*.zip ../OpenPCDet/data/kitti/ && cd ../OpenPCDet/data/kitti/
  unzip xxx.zip
  ```

  ```bash
  OpenPCDet
  ├── data
  │   ├── kitti
  │   │   │── ImageSets
  │   │   │── training
  │   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
  │   │   │── testing
  │   │   │   ├──calib & velodyne & image_2
  ├── pcdet
  ├── tools
  ```

2. 下载数据划分文件

   ```bash
   mkdir ./data/kitti/ && mkdir ./data/kitti/ImageSets

   # 下载数据划分文件
   # val.txt/test.txt/train.txt 原仓库已存在
   # https://github.com/open-mmlab/OpenPCDet/tree/master/data/kitti/ImageSets

   wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/trainval.txt
   ```

3. 划分数据集

   ```bash
   cd OpenPCDet
   python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
   ```

   ```bash
   OpenPCDet
   ├── data
   │   ├── kitti
   │   │   │── gt_database
   │   │   │── ImageSets
   │   │   │── testing
   │   │   │── training
   │   │   │── kitti_dbinfos_train.pkl
   │   │   │── kitti_infos_test.pkl
   │   │   │── kitti_infos_train.pkl
   │   │   │── kitti_infos_trainval.pkl
   │   │   │── kitti_infos_val.pkl
   ```

- 校准数据集：[generate_qdata.py](./generate_qdata.py)

  > 已生成校准数据：[kitti_qdata](https://drive.google.com/drive/folders/1Pt5FsQY9Af6zgEdQrYe0qs1q00XYBxH4?usp=sharing)

  ```bash
  python generate_qdata.py \
  --kitti_data data/kitti/testing/velodyne/ \
  --save_path ./kitti_qdata \
  --quant_num 50 \
  --max_voxel_num 32000 \
  --voxel_size 0.32,0.32,6 \
  --coors_range 0,-39.68,-3,69.12,39.68,1
  ```

- 提取val评估集：[extract_kitti_val_data.py](../source_code/extract_kitti_val_data.py)，将kitti中的val数据集从train文件夹中提取出来

  ```bash
  python extract_kitti_val_data.py \
  --kitti_data /path/to/kitti/ \
  --output_dir ./kitti_val_points
  ```

  - 当前的kitti_val_points中就只包含了val数据集的points数据。

## step.4 模型转换

> 注意此操作在`VACC环境`中执行

1. 在`部署软件整包`中获取安装VAMC模型转换工具

2. 模型转换配置文件

   - [official_kitti_center_point.yaml](../build_in/build/official_kitti_center_point.yaml)
   - 由于硬件的限制，当前只支持int8量化模型，不支持fp16量化模型
   - 转换后将在当前目录下生成`deploy_weights/official_kitti_centerpoint_run_stream_int8`文件夹，其中包含转换后的模型文件

   ```bash
   cd center_point
   mkdir workspace
   cd workspace
   vamc compile ../build_in/build/official_kitti_center_point.yaml
   ```

   > 此处告警`PointPillarScatterFunction`算子是正常的，此算子非onnx标准算子

## step.5 模型推理&评估

1. 参考[vsx/center_point_runstream.py](../build_in/vsx/center_point_runstream.py)：生成推理的txt结果

   ```bash
   python3  ../build_in/vsx/center_point_runstream.py \
       -m "[/path/to/official_kitti_centerpoint_run_stream_int8/mod]" \
       --elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
       --max_voxel_num [32000] \
       --voxel_size [0.32,0.32,6] \
       --coors_range [0,-39.68,-3,69.12,39.68,1] \
       --backbone_input_shape 1,64,496,432 \
       --shuffle_enabled 1 \
       --normalize_enabled 1 \
       --max_points_num 12000000 \
       --dataset_root  /path/to/workspace/kitti_val_points \
       --dataset_output_folder runstream_output
   ```

2. 精度评估

   > 注意此操作在`OpenPCDet环境`中执行

   - 将step5.1中`--dataset_output_folder runstream_output`文件夹拷贝到`OpenPCDet环境`中的相同路径下
   - 需要先修改[kitti_centerpoint_pillar_1x_dataset.yaml](../source_code/config/kitti_centerpoint_pillar_1x_dataset.yaml)文件中的数据集路径DATA_PATH，其路径为/path/to/workspace。这里需要注意确保workspace下存在points/labels/ImageSets文件夹以及custom_infos_train.pkl、custom_infos_train.pkl、custom_dbinfos_train.pkl这几个文件。
   - 评估测评：[eval.py](../source_code/eval.py)
     ```bash
     python ../source_code/eval.py \
         --dataset_yaml ../source_code/config/kitti_centerpoint_pillar_1x_dataset.yaml \
         --result_npz ./runstream_output/ \
         --class_names Car,Truck,Cyclist
     ```

3. 可视化

   > 注意此操作在`桌面环境`中执行

   - 参考[visual.py](../source_code/visual/visual.py)，执行可视化
     ```bash
     python ../source_code/visual/visual.py \
     --task box3d \
     --points_file /path/to/0_1.bin \
     --result_file path/to/runmodel_out/0_1.npz
     ```
     > points_file表示原始点云文件；result_file表示推理结果文件
   - 命令执行后按空格键即可看到点云和检测结果

## step.6 模型推理性能评估

1. 测试最大吞吐

   - 参考[center_point_prof.py](../build_in/vsx/center_point_prof.py)，测试最大吞吐

   ```bash
   python3 ../build_in/vsx/center_point_prof.py \
       -m "[/path/to/official_kitti_centerpoint_run_stream_int8/mod]" \
       --elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
       --max_voxel_num [32000] \
       --max_points_num 2000000 \
       --voxel_size [0.16,0.16,6] \
       --coors_range [0,-39.68,-3,69.12,39.68,1] \
       --shuffle_enabled 1 \
       --normalize_enabled 1 \
       --device_ids [0] \
       --shape [40000] \
       --batch_size 1 \
       --instance 1 \
       --iterations 1500 \
       --input_host 1 \
       --queue_size 1
   ```

2. 测试最小时延

   - 参考[center_point_prof.py](../build_in/vsx/center_point_prof.py)，测试最小时延

   ```bash
   python3 ../build_in/vsx/center_point_prof.py \
       -m "[/path/to/official_kitti_centerpoint_run_stream_int8/mod]" \
       --elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
       --max_voxel_num [32000] \
       --max_points_num 2000000 \
       --voxel_size [0.16,0.16,6] \
       --coors_range [0,-39.68,-3,69.12,39.68,1] \
       --shuffle_enabled 1 \
       --normalize_enabled 1 \
       --device_ids [0] \
       --shape [40000] \
       --batch_size 1 \
       --instance 1 \
       --iterations 1000 \
       --input_host 1 \
       --queue_size 0
   ```

## Appending

1. 如需对照原始pytorch测评
   > 注意此操作在`OpenPCDet环境`中执行
   - 需要把step.1中所有"#export onnx"的注释去掉，还原为之前的代码，然后执行以下命令进行推理。
   ```bash
   cd OpenPCDet/tools
   python test.py \
       --cfg_file /path/to/kitti_centerpoint_pillar_1x_model.yaml \
       --root_path /path/to/workspace \
       --batch_size 8 \
       --ckpt path/to/kitti_centerpoint_pillar_1x.pth
   ```

## Tips

1. 运行时backbone_input_shape参数计算公式

   ```
   伪图尺寸 = 点云各维度range / voxel_size(对应维度)
   伪图尺寸W = (103.6 - (-50)) / 0.32 * 2 = 480
   伪图尺寸H = (50 - (-103.6)) / 0.32 * 2 = 480
   ```

2. 由于点云数据预处理也使用dsp算子进行计算，所以对于最大voxel的数量有一定的限制，需要小于16M

   ```
   最大体素数(max_voxel_num) * 各体素最大点数(max_points_per_voxel) * 数据类型所占字节
   ```

3. 由于VACC部署center_point模型需要把bevfusion后的伪图放到SSRAM中。伪图所占用的空间限制需小于40M

   ```
   伪图宽 x 伪图高 x 通道数 x 数据类型所占字节
   ```
