from trainers.base_trainer import BaseTrainer

import os
import random

import numpy as np
import torch
import time

from tqdm import tqdm

# import networks.cnn.networks as cnet

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from os.path import join as pjoin
import torch.nn.functional as F
import torch.nn as nn

from model.evaluator.losses import InfoNCE_with_filtering


from utils.utils import print_current_loss
from utils.eval_t2m import evaluation_evaluator


class EvaluatorTrainer(BaseTrainer):
    def __init__(self, cfg, eval_model, device):
        self.cfg = cfg
        self.eval_model = eval_model
        self.device = device

        if self.cfg.exp.is_train:
            self.logger = SummaryWriter(self.cfg.exp.log_dir)
            self.l1_criterion_fn = torch.nn.SmoothL1Loss()
            self.latent_loss_fn = torch.nn.SmoothL1Loss()
            self.kl_loss_fn = self.kl_criterion
            self.unit_kl_loss_fn = self.kl_criterion_unit
            self.constrastive_loss_fn = InfoNCE_with_filtering(
                temperature=cfg.training.infoNCE_temp, 
                threshold_selfsim=cfg.training.infoNCE_thre
            )

    def forward(self, batch_data):
        texts, motions, m_lengths = batch_data
        # motions = motions.permute(0, 2, 1).to(self.device).float().detach()
        motions = motions[..., :self.cfg.data.dim_pose].to(self.device).float().detach()
        # motions = motions.permute(0, 2, 1)
        m_lengths = m_lengths.to(self.device).long().detach()

        # print(motions.shape)
        t_latents, t_dists = self.eval_model.encode_text(texts)
        _, m_latents, m_dists = self.eval_model.encode_motion(motions, m_lengths)

        m_motion = self.eval_model.decode(m_latents, motions.shape[1], m_lengths)
        t_motion = self.eval_model.decode(t_latents, motions.shape[1], m_lengths)

        self.t_latents, self.m_latents = t_latents, m_latents
        self.t_dists, self.m_dists = t_dists, m_dists
        self.m_motion, self.t_motion = m_motion, t_motion
        self.motion = motions

    def backward(self):
        # print(self.m_motion.shape, self.motion.shape)
        self.loss_rec_mm = self.l1_criterion_fn(self.m_motion, self.motion)
        self.loss_rec_tm = self.l1_criterion_fn(self.t_motion, self.motion)

        self.unit_kl_t = self.unit_kl_loss_fn(self.t_dists[0], self.t_dists[1])
        self.unit_kl_m = self.unit_kl_loss_fn(self.m_dists[0], self.m_dists[1])
        self.t2m_kl = self.kl_criterion(
            self.t_dists[0], self.t_dists[1], self.m_dists[0], self.m_dists[1]
        )
        self.m2t_kl = self.kl_criterion(
            self.m_dists[0], self.m_dists[1], self.t_dists[0], self.t_dists[1]
        )

        self.latent_align = self.latent_loss_fn(self.t_latents, self.m_latents)
        self.constrative_loss = self.constrastive_loss_fn(
            self.t_latents, self.m_latents, None
        )

        self.losses = (
            (self.loss_rec_mm + self.loss_rec_tm) * self.cfg.training.lambda_rec
            + (self.unit_kl_t + self.unit_kl_m + self.t2m_kl + self.m2t_kl) * self.cfg.training.lambda_kl
            + self.latent_align * self.cfg.training.lambda_latent_align
            + self.constrative_loss * self.cfg.training.lambda_contrast
        )

        loss_logs = OrderedDict({})
        loss_logs["losses"] = self.losses
        loss_logs["loss_rec_mm"] = self.loss_rec_mm
        loss_logs["loss_rec_tm"] = self.loss_rec_tm
        loss_logs["unit_kl_t"] = self.unit_kl_t
        loss_logs["unit_kl_m"] = self.unit_kl_m
        loss_logs["t2m_kl"] = self.t2m_kl
        loss_logs["m2t_kl"] = self.m2t_kl
        loss_logs["latent_align"] = self.latent_align
        loss_logs["constrative_loss"] = self.constrative_loss

        return loss_logs

    def update(self):
        self.zero_grad([self.opt_eval_model])
        loss_logs = self.backward()
        self.losses.backward()

        self.clip_norm([self.eval_model])
                
        self.step(
            [
                self.opt_eval_model,
                self.slr_eval_model
            ]
        )

        return loss_logs

    def save(self, file_name, ep, total_it):

        state = {
            "eval_model": self.eval_model.state_dict(),
            "opt_eval_model": self.opt_eval_model.state_dict(),
            "slr_eval_model": self.slr_eval_model.state_dict(),
            "ep": ep,
            "total_it": total_it,
        }

        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.eval_model.load_state_dict(checkpoint["eval_model"])

        # missing_keys, unexpected_keys = self.eval_model.load_state_dict(checkpoint['eval_model'], strict=False)
        # assert len(unexpected_keys) == 0
        # assert all([k.startswith('T5TextEncoder.') for k in missing_keys])

        if self.cfg.exp.is_train:
            try:
                self.opt_eval_model.load_state_dict(checkpoint["opt_eval_model"])
                self.slr_eval_model.load_state_dict(checkpoint["slr_eval_model"])
            except:
                print("Resume wo optimizer!")
        print(
                "Loading the model from epoch %04d, iteration %06d "
                % (checkpoint["ep"], checkpoint["total_it"])
            )
        return checkpoint["ep"], checkpoint["total_it"]

    def train(self, train_loader, val_loader, eval_val_loader):
        net_list = [self.eval_model]
        self.to(net_list, self.device)

        self.opt_eval_model = optim.AdamW(
            self.eval_model.parameters(),
            betas=(0.9, 0.99),
            lr=self.cfg.training.lr,
            weight_decay=1e-5,
        )

        self.slr_eval_model = optim.lr_scheduler.MultiStepLR(
            self.opt_eval_model, milestones=self.cfg.training.milestones, gamma=self.cfg.training.gamma
        )


        epoch = 0
        it = 0

        if self.cfg.exp.is_continue:
            model_dir = pjoin(self.cfg.exp.model_dir, "latest.tar")
            epoch, it = self.resume(model_dir)
            print("Loading model from epoch %d" % epoch)

        
        _, best_top1, best_top2, best_top3, best_matching = evaluation_evaluator(
                self.cfg.exp.model_dir, eval_val_loader, self.logger, epoch, 0, 
                0, 0, 0, self.eval_model, self.device,
                save_ckpt=False, draw=True
            )

        start_time = time.time()
        total_iters = self.cfg.training.max_epoch * len(train_loader)
        print(
            "Iters Per Epoch, Training: %04d, Validation: %03d"
            % (len(train_loader), len(val_loader))
        )
        min_val_loss = np.inf
        logs = OrderedDict()

        while epoch < self.cfg.training.max_epoch:
            self.net_train(net_list)
            for i, batch_data in enumerate(train_loader):
                if it < self.cfg.training.warm_up_iter:
                    self.update_lr_warm_up(
                        it,
                        self.cfg.training.warm_up_iter,
                        self.cfg.training.lr,
                        [self.opt_eval_model],
                    )

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
                        self.logger.add_scalar(
                            "Train/%s" % tag, value / self.cfg.training.log_every, it
                        )
                        mean_loss[tag] = value / self.cfg.training.log_every
                    self.logger.add_scalar(
                        "Train/lr", self.opt_eval_model.param_groups[0]["lr"], it
                    )
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.cfg.training.save_latest == 0:
                    self.save(pjoin(self.cfg.exp.model_dir, "latest.tar"), epoch, it)

            if epoch % 10 == 0:
                print("Save latest")
                self.save(pjoin(self.cfg.exp.model_dir, "latest.tar"), epoch, it)
            epoch += 1

            print("Validation time:")
            val_loss = None
            with torch.no_grad():
                self.net_eval(net_list)
                for i, batch_data in tqdm(enumerate(val_loader)):
                    # print("Batch %d" % i)
                    self.forward(batch_data)
                    loss_dict = self.backward()
                    if val_loss is None:
                        val_loss = loss_dict
                    else:
                        for k, v in loss_dict.items():
                            val_loss[k] += v

            print_str = "Validation Loss:"

            for tag, value in val_loss.items():
                val_loss[tag] /= len(val_loader)
                print_str += " %s: %.4f " % (tag, val_loss[tag])

                self.logger.add_scalar("Val/%s" % tag, val_loss[tag], epoch)

            # self.logger.add_scalar("Val/loss", val_loss["loss"], epoch)
            print(print_str)

            if val_loss["losses"] < min_val_loss:
                min_val_loss = val_loss["losses"]
                # min_val_epoch = epoch
                print("Best Validation Model So Far!~")
                self.save(pjoin(self.cfg.exp.model_dir, "best.tar"), epoch, it)
            
            _, best_top1, best_top2, best_top3, best_matching = evaluation_evaluator(
                self.cfg.exp.model_dir, eval_val_loader, self.logger, epoch, best_top1, 
                best_top2, best_top3, best_matching, self.eval_model, self.device,
                save_ckpt=True, draw=True
            )
