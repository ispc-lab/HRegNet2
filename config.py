import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser(description='HRegNet')

# Training
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--runname', type=str, default='')
parser.add_argument('--augment', type=float, default=0.0)
parser.add_argument('--ckpt_dir', type=str, default='')
parser.add_argument('--freeze_detector', type=str2bool, default=True)
parser.add_argument('--freeze_feats', type=str2bool, default=True)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--stat_freq', type=int, default=100)
parser.add_argument('--val_epoch_freq', type=int, default=1)

# Data
parser.add_argument('--root', type=str, default='')
parser.add_argument('--npoints', type=int, default=16384)
parser.add_argument('--voxel_size', type=float, default=0.3)
parser.add_argument('--dataset', type=str, default='kitti')
parser.add_argument('--data_list', type=str, default='')
parser.add_argument('--pretrain_feats', type=str, default=None)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--pretrain_weights', type=str, default=None)

# Model
parser.add_argument('--use_fps', type=str2bool, default=True)
parser.add_argument('--use_weights', type=str2bool, default=True)
parser.add_argument('--K_C', type=int, default=4)
parser.add_argument('--K_N', type=int, default=8)

# Others
parser.add_argument('--trans_thresh', type=float, default=2.0)
parser.add_argument('--rot_thresh', type=float, default=5.0)
parser.add_argument('--save_dir', type=str, default=None)

def get_config():
    args = parser.parse_args()
    return args