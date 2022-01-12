import torch
import os
import torch.nn as nn
import numpy as np
import torch.optim as optim

from models.models import HRegNet
from models.utils import get_logger, ensure_dir, calc_error_np, AverageMeter
from models.losses import transformation_loss

from tqdm import tqdm

import shutil

class RegTrainer:
    def __init__(self, config, train_loader, val_loader):

        self.config = config
        self.max_epoch = self.config.epochs
        self.checkpoint_dir = os.path.join(self.config.ckpt_dir, self.config.runname)
        ensure_dir(self.checkpoint_dir)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.best_val_succ = 0.0
        self.best_val_epoch = 0
        self.best_val_rte = 1000.0

        self.model = HRegNet(self.config).cuda()
        self.model.feature_extraction.load_state_dict(torch.load(self.config.pretrain_feats))

        log_path = os.path.join(self.checkpoint_dir, "train.log")
        self.logger = get_logger(log_path)

        snapshot_dir = os.path.join(self.checkpoint_dir, 'snapshot')
        ensure_dir(snapshot_dir)

        BASE_DIR = os.getcwd()
        shutil.copy(os.path.join(BASE_DIR, 'models/layers.py'), \
            os.path.join(os.path.join(snapshot_dir, 'layers.py')))
        shutil.copy(os.path.join(BASE_DIR, 'models/models.py'), \
            os.path.join(os.path.join(snapshot_dir, 'models.py')))
        shutil.copy(os.path.join(BASE_DIR, 'models/trainer.py'), \
            os.path.join(os.path.join(snapshot_dir, 'trainer.py')))
        shutil.copy(os.path.join(BASE_DIR, 'scripts/train_reg.sh'), \
            os.path.join(os.path.join(snapshot_dir, 'train_reg.sh')))
        
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config.lr,
                                    betas=(0.9, 0.999)
                                    )
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.start_epoch = 0

        if config.resume is not None:
            self.logger.info('Resume training from ' + config.resume)
            state_dict = torch.load(config.resume)
            self.model.load_state_dict(state_dict['state_dict'])
            self.start_epoch = state_dict['epoch']
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])
    
    def train(self):
        self.logger.info(
            f'Current best val model with {self.best_val_succ} at epoch {self.best_val_epoch}'
        )

        for epoch in range(self.start_epoch+1, self.max_epoch+1):

            lr = self.scheduler.get_lr()
            
            self.logger.info(f"Epoch:{epoch}, LR: {lr}")

            train_loss, train_trans_loss, train_rot_loss = self.train_one_epoch(epoch)

            self.scheduler.step()

            self.save_ckpt(epoch, 'checkpoint')

            if epoch % self.config.val_epoch_freq == 0:
                val_succ, val_rte, val_rre = self.val_one_epoch(epoch)

                if val_rte <= self.best_val_rte:
                    self.best_val_rte = val_rte
                    self.best_val_epoch = epoch
                    self.save_ckpt(epoch, 'best_val_checkpoint')

                    self.logger.info(
                        f'Save the best val model at epoch {epoch} with succ {self.best_val_rte}'
                    )
                else:
                    self.logger.info(
                        f'Current best val model with {self.best_val_rte} at epoch {self.best_val_epoch}'
                    )
    
    def train_one_epoch(self, epoch):

        self.model.train()
        total_loss, total_num = 0, 0.0
        self.train_loader.dataset.reset_seed(0)
        data_loader = self.train_loader

        num_train_iter = len(data_loader)

        loss_meter = AverageMeter()
        trans_loss_meter = AverageMeter()
        rot_loss_meter = AverageMeter()

        epoch_loss_meter = AverageMeter()
        epoch_trans_loss_meter = AverageMeter()
        epoch_rot_loss_meter = AverageMeter()

        trainer_loader_iter = data_loader.__iter__()

        for iter in range(num_train_iter):

            src_points, dst_points, gt_R, gt_t = trainer_loader_iter.next()
            src_points = src_points.cuda()
            dst_points = dst_points.cuda()
            gt_R = gt_R.cuda()
            gt_t = gt_t.cuda()

            self.optimizer.zero_grad()
            ret_dict = self.model(src_points, dst_points)

            l_trans = 0.0
            l_R = 0.0
            l_t = 0.0
            for idx in range(3):
                l_trans_, l_R_, l_t_ = transformation_loss(ret_dict['rotation'][idx], ret_dict['translation'][idx], gt_R, gt_t, self.config.alpha)
                l_trans += l_trans_
                l_R += l_R_
                l_t += l_t_

            l_trans = l_trans / 3.0
            l_R = l_R / 3.0
            l_t = l_t / 3.0
            
            loss = l_trans
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            with torch.no_grad():

                loss_meter.update(loss.item())
                trans_loss_meter.update(l_t.item())
                rot_loss_meter.update(l_R.item())

                epoch_loss_meter.update(loss.item())
                epoch_trans_loss_meter.update(l_t.item())
                epoch_rot_loss_meter.update(l_R.item())
            
            if iter % self.config.stat_freq == 0:
                self.logger.info(' '.join([
                    f"Train Epoch: {epoch} [{iter}/{num_train_iter}],",
                    f"Loss: {loss_meter.avg:.3e},",
                    f"Translation Loss: {trans_loss_meter.avg:.3e},",
                    f"Rotation Loss: {rot_loss_meter.avg:.3e}"
                ]))
                
                loss_meter.reset()
                trans_loss_meter.reset()
                rot_loss_meter.reset()
        
        self.logger.info(' '.join([
                    f"Average stat of Epoch: {epoch},",
                    f"Loss: {epoch_loss_meter.avg:.3e},",
                    f"Translation loss: {epoch_trans_loss_meter.avg:.3e}",
                    f"Rotation loss: {epoch_rot_loss_meter.avg:3e}"
                ]))
        
        return epoch_loss_meter.avg, epoch_trans_loss_meter.avg, epoch_rot_loss_meter.avg
    
    def val_one_epoch(self, epoch):

        self.logger.info(f'Start validating Epoch: {epoch}')
        self.model.eval()
        total_loss = 0
        total_R_loss = 0
        total_t_loss = 0
        
        trans_error_list = []
        rot_error_list = []
        valid_trans_error_list = []
        valid_rot_error_list = []

        valid_count = 0
        total_count = 0

        self.val_loader.dataset.reset_seed(0)
        data_loader = self.val_loader

        num_val_iter = len(data_loader)

        val_loader_iter = data_loader.__iter__()

        for iter in tqdm(range(num_val_iter), ncols=80):

            with torch.no_grad():
                src_points, dst_points, gt_R, gt_t = val_loader_iter.next()
                src_points = src_points.cuda()
                dst_points = dst_points.cuda()
                gt_R = gt_R.cuda()
                gt_t = gt_t.cuda()

                ret_dict = self.model(src_points, dst_points)
                l_trans, l_R, l_t = transformation_loss(ret_dict['rotation'][-1], ret_dict['translation'][-1], gt_R, gt_t, self.config.alpha)
                total_loss += l_trans.item()
                total_R_loss += l_R.item()
                total_t_loss += l_t.item()

                pred_R = ret_dict['rotation'][-1]
                pred_t = ret_dict['translation'][-1]

                for idx in range(pred_R.shape[0]):

                    pred_R_ = pred_R[idx,:,:].squeeze().cpu().numpy()
                    pred_t_ = pred_t[idx,:].squeeze().cpu().numpy()
                    rot_error, trans_error = calc_error_np(pred_R_, pred_t_, \
                        gt_R[idx,:,:].squeeze().cpu().numpy(), gt_t[idx,:].squeeze().cpu().numpy())
                    
                    trans_error_list.append(trans_error)
                    rot_error_list.append(rot_error)
                    total_count += 1

                    if trans_error < self.config.trans_thresh and rot_error < self.config.rot_thresh:
                        valid_trans_error_list.append(trans_error)
                        valid_rot_error_list.append(rot_error)
                        valid_count += 1
                
        total_loss = total_loss/num_val_iter
        total_R_loss = total_R_loss/num_val_iter
        total_t_loss = total_t_loss/num_val_iter

        success_rate = valid_count / total_count
        trans_error_array = np.array(valid_trans_error_list)
        rot_error_array = np.array(valid_rot_error_list)
        trans_mean = np.mean(trans_error_array)
        trans_std = np.std(trans_error_array)
        rot_mean = np.mean(rot_error_array)
        rot_std = np.std(rot_error_array)

        self.logger.info(' '.join([
                    f"Average val stat of Epoch: {epoch},",
                    f"Loss: {total_loss:.3e},",
                    f"Trans error: {trans_mean:.4f}+-{trans_std:.4f}",
                    f"Rot error: {rot_mean:.4f}+-{rot_std:.4f}",
                    f"Succ rate: {success_rate:3e}"
                    ]))
        
        return success_rate, trans_mean, rot_mean
    
    def save_ckpt(self, epoch, filename='checkpoint'):

        self.logger.info(f'Save checkpoint for epoch {epoch}.')
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
        torch.save(state, filename)