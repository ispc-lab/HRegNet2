import os
import numpy as np
from numpy.matrixlib.defmatrix import _convert_from_string
import torch
from config import get_config

from data.kitti_data import KittiDataset
from data.nuscenes_data import NuscenesDataset
from data.apollo_data import ApolloDataset

from models.models import HRegNet
from models.utils import calc_error_np, set_seed
from tqdm import tqdm
import datetime

def test(args):
    if args.dataset == 'kitti':
        test_seqs = ['08','09','10']
        test_dataset = KittiDataset(args.root, test_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)
    elif args.dataset == 'nusc':
        test_seqs = ['test']
        test_dataset = NuscenesDataset(args.root, test_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)
    elif args.dataset == 'apollo':
        test_seqs = ['test']
        data_root = os.path.join(args.root, 'TestData')
        test_dataset = ApolloDataset(data_root, test_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)
    else:
        raise('Not implemented')
    
    net = HRegNet(args).cuda()
    net.load_state_dict(torch.load(args.pretrain_weights)['state_dict'])
    net.eval()

    trans_error_list = []
    rot_error_list = []
    pred_T_list = []
    delta_t_list = []
    trans_thresh = 2.0
    rot_thresh = 5.0

    with torch.no_grad():
        for idx in tqdm(range(test_dataset.__len__()), ncols=80):
            
            src_points, dst_points, gt_R, gt_t = test_dataset[idx]
            src_points = src_points.unsqueeze(0).cuda()
            dst_points = dst_points.unsqueeze(0).cuda()
            gt_R = gt_R.numpy()
            gt_t = gt_t.numpy()
            start_t = datetime.datetime.now()
            ret_dict = net(src_points, dst_points)
            end_t = datetime.datetime.now()
            pred_R = ret_dict['rotation'][-1]
            pred_t = ret_dict['translation'][-1]
            pred_R = pred_R.squeeze().cpu().numpy()
            pred_t = pred_t.squeeze().cpu().numpy()
            rot_error, trans_error = calc_error_np(pred_R, pred_t, gt_R, gt_t)
            
            pred_T = np.zeros((3,4))
            gt_T = np.zeros((3,4))
            pred_T[:3,:3] = pred_R
            pred_T[:3,3] = pred_t
            gt_T[:3,:3] = gt_R
            gt_T[:3,3] = gt_t
            pred_T = pred_T.flatten()
            gt_T = gt_T.flatten()
            pred_T_list.append(pred_T)
            
            if trans_error < trans_thresh and rot_error < rot_thresh:
                trans_error_list.append(trans_error)
                rot_error_list.append(rot_error)
            
            delta_t = (end_t - start_t).microseconds
            delta_t_list.append(delta_t)
    
    success_rate = len(trans_error_list)/test_dataset.__len__()
    trans_error_array = np.array(trans_error_list)
    rot_error_array = np.array(rot_error_list)
    trans_mean = np.mean(trans_error_array)
    trans_std = np.std(trans_error_array)
    rot_mean = np.mean(rot_error_array)
    rot_std = np.std(rot_error_array)
    delta_t_array = np.array(delta_t_list)
    delta_t_mean = np.mean(delta_t_array)

    print('Translation mean: {:.4f}'.format(trans_mean))
    print('Translation std: {:.4f}'.format(trans_std))
    print('Rotation mean: {:.4f}'.format(rot_mean))
    print('Rotation std: {:.4f}'.format(rot_std))
    print('Runtime: {:.4f}'.format(delta_t_mean))
    print('Success rate: {:.4f}'.format(success_rate))
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    pred_T_array = np.array(pred_T_list)
    np.savetxt(os.path.join(args.save_dir, args.dataset+'_pred.txt'), pred_T_array)
    np.savetxt(os.path.join(args.save_dir, args.dataset+'_trans_error.txt'), trans_error_list)
    np.savetxt(os.path.join(args.save_dir, args.dataset+'_rot_error.txt'), rot_error_list)

    f_summary = open(os.path.join(args.save_dir, args.dataset+'_summary.txt'), 'w')
    f_summary.write('Dataset: '+args.dataset+'\n')
    f_summary.write('Translation threshold: {:.2f}\n'.format(trans_thresh))
    f_summary.write('Rotation threshold: {:.2f}\n'.format(rot_thresh))
    f_summary.write('Translation mean: {:.4f}\n'.format(trans_mean))
    f_summary.write('Translation std: {:.4f}\n'.format(trans_std))
    f_summary.write('Rotation mean: {:.4f}\n'.format(rot_mean))
    f_summary.write('Rotation std: {:.4f}\n'.format(rot_std))
    f_summary.write('Runtime: {:.4f}\n'.format(delta_t_mean))
    f_summary.write('Success rate: {:.4f}\n'.format(success_rate))
    f_summary.close()

    print('Saved results to ' + args.save_dir)

if __name__ == '__main__':
    config = get_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    set_seed(config.seed)

    test(config)