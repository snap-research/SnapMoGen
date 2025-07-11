from trainers.base_trainer import BaseTrainer

import os
import random

import numpy as np
import torch
import time

from tqdm import tqdm

# import networks.cnn.networks as cnet

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter # type: ignore
from collections import OrderedDict
from os.path import join as pjoin


from utils.utils import print_current_loss


def length_to_mask(length, max_len, device: torch.device = None) -> torch.Tensor:
    if device is None:
        device = length.device

    if isinstance(length, list):
        length = torch.tensor(length)
    
    length = length.to(device)
    # max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask


def mean_flat(tensor: torch.Tensor, mask=None):
    """
    Take the mean over all non-batch dimensions.
    """
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        # mask = mask.unsqueeze(2)           # [B, T] -> [T, B, 1]
        assert tensor.dim() == 3
        denom = mask.sum() * tensor.shape[-1]
        loss = (tensor * mask).sum() / denom
        return loss
    

class GlobalRegressorTrainer(BaseTrainer):
    def __init__(self, cfg, regressor, device):
        self.cfg = cfg
        self.device = device
        self.regressor = regressor
        # self.latent_dec = latent_dec

        if self.cfg.exp.is_train:
            self.logger = SummaryWriter(self.cfg.exp.log_dir)
            self.l1_criterion = torch.nn.SmoothL1Loss()
    

    def forward(self, batch_data):
        M = batch_data
        M = M.permute.to(self.device).float().detach()

        # M1 = M1[:, 3:-4]
        # M2 = M2[:, 3:-4]
        noise = torch.randn_like(M) * random.random() * 0.1 
        # torch.rand
        input = noise + M
        # print(input.shape)
        model_input = torch.cat([input[..., 0:1], input[..., 3:self.cfg.data.dim_pose]], dim=1)

        pred = self.regressor(model_input)
        

        self.M, self.pred = M, pred
        self.input = input

    
    def backward(self):
        # Reconstruction loss
        self.loss = self.l1_criterion(self.M[..., 1:3], self.pred)
        loss_logs = OrderedDict({})
        
        loss_logs["loss"] = self.loss.item()

        return loss_logs

    def update(self):
        self.zero_grad([self.opt_regressor])
        loss_logs = self.backward()
        self.loss.backward()
        self.clip_norm([self.regressor])
        self.step([self.opt_regressor, self.slr_regressor])
        # self.scheduler.step()
        return loss_logs
    
    def save(self, file_name, ep, total_it):
        state = {
            "regressor": self.regressor.state_dict(),

            "opt_regressor": self.opt_regressor.state_dict(),
            "slr_regressor":self.slr_regressor.state_dict(),

            "ep": ep,
            "total_it": total_it,
        }

        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.regressor.load_state_dict(checkpoint["regressor"])

        if self.cfg.exp.is_train:
            self.opt_regressor.load_state_dict(checkpoint["opt_regressor"])
            self.slr_regressor.load_state_dict(checkpoint["slr_regressor"])
        print("Loading the model from epoch %04d, iteration %06d "%(checkpoint["ep"], checkpoint["total_it"]))
        return checkpoint["ep"], checkpoint["total_it"]
    
    def train(self, train_loader, val_loader, plot_eval):
        net_list = [self.regressor]
        self.to(net_list, self.device)

        self.opt_regressor = optim.AdamW(self.regressor.parameters(), betas=(0.9, 0.99), lr=self.cfg.training.lr, weight_decay=1e-5) # type: ignore
        self.slr_regressor = optim.lr_scheduler.MultiStepLR(self.opt_regressor,
                                                        milestones=self.cfg.training.milestones,
                                                        gamma=self.cfg.training.gamma)

        epoch = 0
        it = 0

        if self.cfg.exp.is_continue:
            model_dir = pjoin(self.cfg.exp.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Loading model from epoch %d" % epoch)
        
        start_time = time.time()
        total_iters = self.cfg.training.max_epoch * len(train_loader)
        print("Iters Per Epoch, Training: %04d, Validation: %03d" % (len(train_loader), len(val_loader)))
        min_val_loss = np.inf
        logs = OrderedDict()

        while epoch < self.cfg.training.max_epoch:                
            self.net_train(net_list)
            for i, batch_data in enumerate(train_loader):
                if it < self.cfg.training.warm_up_iter:
                    self.update_lr_warm_up(it, self.cfg.training.warm_up_iter, self.cfg.training.lr, [self.opt_regressor])

                self.forward(batch_data)
                loss_dict = self.update()

                for k, v in loss_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v
                it += 1
                if it % self.cfg.training.log_every == 0:
                    mean_loss = OrderedDict()

                    for tag, value in logs.items():
                        self.logger.add_scalar("Train/%s"%tag, value / self.cfg.training.log_every, it)
                        mean_loss[tag] = value / self.cfg.training.log_every
                    self.logger.add_scalar("Train/lr", self.opt_regressor.param_groups[0]['lr'], it)
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.cfg.training.save_latest == 0:
                    self.save(pjoin(self.cfg.exp.model_dir, 'latest.tar'), epoch, it)

            # if epoch%10 == 0:
            #     print("Save latest")
            self.save(pjoin(self.cfg.exp.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            print("Validation time:")
            val_loss = {}
            with torch.no_grad():
                self.net_eval(net_list)
                for i, batch_data in enumerate(val_loader):
                    if i % 100 == 0:
                        print("Batch %d"%i)
                    self.forward(batch_data)
                    loss_dict = self.backward()
                    if val_loss is None:
                        val_loss = loss_dict
                    else:
                        for k, v in loss_dict.items():
                            val_loss[k] += v

            print_str = "Validation Loss:"

            for tag, value in val_loss.items(): # type: ignore
                val_loss[tag] /= len(val_loader)
                print_str += ' %s: %.4f ' % (tag, val_loss[tag])

                self.logger.add_scalar("Val/%s"%tag, val_loss[tag], epoch)

            # self.logger.add_scalar("Val/loss", val_loss["loss"], epoch)
            print(print_str)

            if val_loss["loss"] < min_val_loss:
                min_val_loss = val_loss["loss"]
                # min_val_epoch = epoch
                print("Best Validation Model So Far!~")
                self.save(pjoin(self.cfg.exp.model_dir, "best.tar"), epoch, it)

            if epoch % self.cfg.training.eval_every_e == 0:
                # B = self.M.size(0)
                # print(self.M2[:6:2, :3].shape, self.RM2[:6:2].shape)
                RM = torch.cat([self.M[:4:2,:, 0:1], self.pred[:4:2], self.M[:4:2, :,3:]], dim=1)
                RMM = torch.cat([self.input[:4:2,:, 0:1], self.pred[:4:2], self.input[:4:2,:, 3:]], dim=1)
                data = torch.cat([self.M[:4:2], RM, RMM], dim=0)
                # data = data.permute(0, 2, 1).detach().cpu()
                save_dir = pjoin(self.cfg.exp.eval_dir, "E%04d" % (epoch))
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(data, save_dir)