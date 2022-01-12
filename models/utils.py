import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F

import point_utils_cuda
import logging
import coloredlogs
import os

import random

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        '''
        ctx:
        xyz: [B,N,3]
        npoint: int
        '''
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        point_utils_cuda.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output
    
    @staticmethod
    def backward(xyz, a=None):
        return None, None

furthest_point_sample = FurthestPointSampling.apply

class WeightedFurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, weights: torch.Tensor, npoint: int) -> torch.Tensor:
        '''
        ctx:
        xyz: [B,N,3]
        weights: [B,N]
        npoint: int
        '''
        assert xyz.is_contiguous()
        assert weights.is_contiguous()
        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        point_utils_cuda.weighted_furthest_point_sampling_wrapper(B, N, npoint, xyz, weights, temp, output);
        return output
    
    @staticmethod
    def backward(xyz, a=None):
        return None, None

weighted_furthest_point_sample = WeightedFurthestPointSampling.apply

class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        '''
        ctx
        features: [B,C,N]
        idx: [B,npoint]
        '''
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint)

        point_utils_cuda.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output
    
    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()
        grad_features = Variable(torch.cuda.FloatTensor(B,C,N).zero_())
        grad_out_data = grad_out.data.contiguous()
        point_utils_cuda.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None

gather_operation = GatherOperation.apply

def set_seed(seed):
    '''
    Set random seed for torch, numpy and python
    '''
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed) 
        
    torch.backends.cudnn.benchmark=False 
    torch.backends.cudnn.deterministic=True

def get_logger(log_path):

    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    file_handler = logging.FileHandler(log_path)
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info('Output and logs will be saved to {}'.format(log_path))

    return logger

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calc_error_np(pred_R, pred_t, gt_R, gt_t):
    tmp = (np.trace(pred_R.transpose().dot(gt_R))-1)/2
    if np.abs(tmp) > 1.0:
        tmp = 1.0
    L_rot = np.arccos(tmp)
    L_rot = 180 * L_rot / np.pi
    L_trans = np.linalg.norm(pred_t - gt_t)
    return L_rot, L_trans

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0
    
    def update(self, val, n=1):
        if isinstance(val, np.ndarray):
            n = val.size
            val = val.mean()
        elif isinstance(val, torch.Tensor):
            n = val.nelement()
            val = val.mean().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val**2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2