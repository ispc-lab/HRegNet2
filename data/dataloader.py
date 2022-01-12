from data.kitti_data import KittiDataset
from data.nuscenes_data import NuscenesDataset
from data.apollo_data import ApolloDataset

from torch.utils.data import DataLoader

import os

def make_data_loader(args, phase=None):

    if args.dataset == 'kitti':

        val_seqs = ['06','07']
        val_dataset = KittiDataset(args.root, val_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)

        test_seqs = ['08','09','10']
        test_dataset = KittiDataset(args.root, test_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)

        train_seqs = ['00','01','02','03','04','05']
        train_dataset = KittiDataset(args.root, train_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)

    elif args.dataset == 'nusc':
        val_seqs = ['val']
        val_dataset = NuscenesDataset(args.root, val_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)

        test_seqs = ['test']
        test_dataset = NuscenesDataset(args.root, test_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)

        train_seqs = ['train']
        train_dataset = NuscenesDataset(args.root, train_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)
    elif args.dataset == 'apollo':
        val_seqs = ['val']
        data_root = os.path.join(args.root, 'TrainData')
        val_dataset = ApolloDataset(data_root, val_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)

        test_seqs = ['test']
        data_root = os.path.join(args.root, 'TestData')
        test_dataset = ApolloDataset(data_root, test_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)

        test_seqs = ['train']
        data_root = os.path.join(args.root, 'TrainData')
        train_dataset = ApolloDataset(data_root, test_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)

    else:
        raise('Not implemented')
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True)
    
    return train_loader, val_loader, test_loader
