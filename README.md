## HRegNet: A Hierarchical Network for Efficient and Accurate Outdoor LiDAR Point Cloud Registration

### Environments
The code mainly requires the following libraries and you can check `requirements.txt` for more environment requirements.
- PyTorch 1.7.0/1.7.1
- Cuda 11.0/11.1
- [pytorch3d 0.6.0](https://github.com/facebookresearch/pytorch3d)
- [open3d 0.13.0](http://www.open3d.org)

Please run the following commands to install `point_utils`
```
cd model/PointUtils
python setup.py install
```

**Training device**: NVIDIA RTX 3090

### Datasets
The point cloud pairs list and the ground truth relative transformation are stored in `data/kitti_list`, `data/nuscenes_list` and `data/apollo_list`. 
The data of the three datasets should be organized as follows:
#### KITTI odometry dataset
```
DATA_ROOT
├── 00
│   ├── velodyne
│   ├── calib.txt
├── 01
├── ...
```
#### NuScenes dataset
```
DATA_ROOT
├── v1.0-trainval
│   ├── maps
│   ├── samples
│   │   ├──LIDAR_TOP
│   ├── sweeps
│   ├── v1.0-trainval
├── v1.0-test
│   ├── maps
│   ├── samples
│   │   ├──LIDAR_TOP
│   ├── sweeps
│   ├── v1.0-test
```
#### Apollo-SouthBay dataset
```
DATA_ROOT
├── TrainData
│   ├── BaylandsToSeafood
│   ├── ColumbiaPark
│   ├── ...
├── TestData
│   ├── BaylandsToSeafood
│   ├── ColumbiaPark
│   ├── ...
```

### Train
We provide training scripts in `scripts/`.

Please specify the following entries:
- `DATASET`: ['kitti','nusc','apollo']
- `ROOT`: Root of the dataset
- `DATA_LIST`: Data list in `data/data_list`, e.g., `data/data_list/kitti_list`
- `CKPT_DIR`: The dir you want to save the ckpt and log files
- `NPOINTS`: 16384 for kitti and apollo, 8192 for nuscenes
- `pretrain_feats`: Pretrain weights for feature extractor
- `GPU`: GPU Id if you have multiple GPUs


### Test
We provide pre-trained weights for three datasets in `ckpt/pretrained/kitti_release/`, `ckpt/pretrained/nusc_release/` and `ckpt/pretrained/apollo_release/`, respectively. And the test scripts are provided in `scripts/`. 

Please specify the following entries:
- `DATASET`: ['kitti','nusc','apollo']
- `ROOT`: Root of the dataset
- `DATA_LIST`: Data list in `data/data_list`, e.g., `data/data_list/kitti_list`
- `SAVE_DIR`: The dir you want to save the results
- `PRETRAIN_WEIGHTS`: Pretrain weights in `ckpt/pretrained`, e.g., `ckpt/pretrained/kitti_release/kitti.pth`
- `NPOINTS`: 16384 for kitti and apollo, 8192 for nuscenes

