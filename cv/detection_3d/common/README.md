# common tools

## kitti_eval
> 基于kitti自带的eval方法，进行精度评估

1. 获取KITTI数据集：[KITTI 3D object detection dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

- 所需数据如下：

  - 彩色图像数据（12GB）， data_object_image_2.zip
  - 点云数据（29GB），data_object_velodyne.zip
  - 相机矫正数据（16MB），data_object_calib.zip
  - 标签数据（5MB），data_object_label_2.zip

  > 其中彩色图像数据、点云数据、相机矫正数据均包含training（7481）和testing（7518）两个部分，标签数据只在training数据中。

  ```bash
  mv data*.zip /path/to/OpenPCDet/data/kitti/ && cd /path/to/OpenPCDet/data/kitti/
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
   # 下载数据划分文件
   # val.txt/test.txt/train.txt 原仓库已存在
   # https://github.com/open-mmlab/OpenPCDet/tree/master/data/kitti/ImageSets
   cd /path/to/OpenPCDet
   wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/trainval.txt
   ```

3. 划分数据集

   ```bash
   cd /path/to/OpenPCDet
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
4. 数据预处理主要是从原始数据中提取并处理视野(Field of View, FOV)内的点云数据。参考[get_fov_data.py](./kitti_eval/get_fov_data.py)

   ```bash
   cd ./kitti_eval/
   python get_fov_data.py \
   --dataset_yaml /path/to/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml \
   --data_path /path/to/OpenPCDet/data/kitti
   ```

   ```bash
   #主要生成了val文件
   OpenPCDet
   ├── data
   │   ├── kitti
   │   │   │── val
   │   │   │   ├── calib & label_2
   │   │   │   ├── fov_pointcloud_float32
   │   │   │   ├── fov_pointcloud_float16
   ```

5. 相关代码：[kitti_eval](./kitti_eval)


6. 精度测评
- 按前文准备数据步骤，移动val文件夹下生成calib和label_2文件夹数据
    > 将`calib`文件夹移动到`./kitti_eval/evals`文件夹下，`label_2`移动到`./kitti_eval/kitti_eval_system`文件夹下

    ```bash
    cd ./kitti_eval

    ln -s  /path/to/OpenPCDet/data/kitti/val/calib evals/kitti
    ln -s  /path/to/OpenPCDet/data/kitti/val/label_2 kitti_eval_system/label
    ln -s  /path/to/OpenPCDet/data/kitti/kitti_infos_val.pkl evals/kitti
    ```

- 该方法的原理是将pointpillar模型的输出结果，通过相机内参等信息，通过各种变换转换成kitti标注txt的形式，然后再计算精度等结果。
    - kitti的标签文件每行数据代表的意思分别是：类别/截断程度/遮挡等级/物体观察角度/2D边界框坐标(x1,y1,x2,y2)/3D尺寸(h,w,l)/3D位置(x,y,z)/绕相机Y轴的旋转角，总共有15个值
    - 模型输出结果通过相机内参信息转换后，生成的txt文件内容中除了上述15个值之外，后面会有一个confidence的值，总共16个值
- 该方法需要保证**每个点云文件都有对应的相机内参信息**时才能使用。
    - kitti数据集中，每个点云文件对应的相机内参信息在kitti的calib文件夹中，每个点云文件对应一个calib文件，文件名与点云文件名相同
- [evaluation.py](./kitti_eval/evaluation.py)：进行精度评估
    ```
    python ./evaluation.py \
    --out_dir model_infer_output  # 此处指向模型推理结果路径
    ```