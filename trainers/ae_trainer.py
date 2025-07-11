from trainers.base_trainer import BaseTrainer

import os

import torch
import time

from copy import deepcopy

# import networks.cnn.networks as cnet

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict, defaultdict
from os.path import join as pjoin


from utils.utils import print_current_loss, print_val_loss
from utils.eval_t2m import evaluation_vqvae, evaluation_vqvae_hml


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


class VQTokenizerTrainer(BaseTrainer):
    def __init__(self, cfg, vq_model, device):
        self.cfg = cfg
        self.vq_model = vq_model
        self.device = device

        if cfg.exp.is_train:
            self.logger = SummaryWriter(cfg.exp.log_dir)
            if cfg.training.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss()
            elif cfg.training.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss()

        if cfg.training.ema:
            self.ema_model = deepcopy(vq_model).to(device)
            self.ema_model.eval()
            self.requires_grad(self.ema_model, False)

        # self.critic = CriticWrapper(self.cfg.dataset_name, self.cfg.device)

    def forward(self, batch_data, vq_model, fk_func):
        motions = batch_data.detach().to(self.device).float()
        pred_motion, loss_commit, perplexity = vq_model(motions[..., :self.cfg.data.dim_pose])
        
        self.motions = motions[..., :self.cfg.data.dim_pose]
        self.pred_motion = pred_motion

        loss_rec = self.l1_criterion(self.pred_motion, self.motions)
        # pred_local_pos = pred_motion[..., 4 : self.cfg.data.joint_num * 3 + 4]
        # local_pos = motions[..., 4 : self.cfg.data.joint_num * 3 + 4]
        if self.cfg.training.lambda_fk == 0:
            loss_fk = self.l1_criterion(motions, motions)
        else:
            loss_fk = self.l1_criterion(fk_func(self.motions), fk_func(self.pred_motion))

        loss_global = self.l1_criterion(self.pred_motion[..., :4], self.motions[..., :4])
        loss_vel = self.l1_criterion(self.pred_motion[:, 1:]-self.pred_motion[:, :-1],
                                     self.motions[:, 1:]-self.motion[:, :-1])

        loss = loss_rec + \
            self.cfg.training.lambda_global * loss_global + \
            self.cfg.training.lambda_fk * loss_fk + \
                self.cfg.training.lambda_commit * loss_commit

        # return loss, loss_rec, loss_vel, loss_commit, perplexity
        # return loss, loss_rec, loss_percept, loss_commit, perplexity
        loss_logs = OrderedDict()
        loss_logs["loss"] = loss.item()
        loss_logs["loss_rec"] = loss_rec.item()
        loss_logs["loss_global"] = loss_global.item()
        loss_logs["loss_commit"] = loss_commit.item()
        loss_logs["loss_fk"] = loss_fk.item()
        loss_logs["loss_vel"] = loss_vel.item()
        loss_logs["perplexity"] = perplexity.item()
        return loss, loss_logs
    

    def forward_attn(self, batch_data, vq_model, fk_func):
        _, motions, m_lens = batch_data
        motions = motions.detach().to(self.device).float()
        m_lens = m_lens.detach().to(self.device).long()
        pred_motion, loss_commit, perplexity = vq_model(motions[..., :self.cfg.data.dim_pose], m_lens.clone())
        
        self.motions = motions[..., :self.cfg.data.dim_pose]
        self.pred_motion = pred_motion

        # loss_rec = self.l1_criterion(self.pred_motion, self.motions)
        mask = length_to_mask(m_lens, max_len=motions.shape[1])
        loss_rec = mean_flat(
            F.smooth_l1_loss(self.pred_motion, self.motions, reduction='none'),
            mask=mask.unsqueeze(-1)
        )

        
        if self.cfg.training.lambda_fk == 0:
            loss_fk = self.l1_criterion(motions, motions)
        else:
            B, T, _ = motions.shape
            loss_fk = mean_flat(
                F.smooth_l1_loss(fk_func(self.motions).view(B, T, -1), 
                                 fk_func(self.pred_motion).view(B, T, -1), 
                                 reduction='none'),
                mask=mask.unsqueeze(-1))

        loss_global = mean_flat(
            F.smooth_l1_loss(self.pred_motion[..., :4], self.motions[..., :4], reduction='none'),
            mask=mask.unsqueeze(-1)
        )

        if self.cfg.data.name == 'snapmogen':
            loss_vel = mean_flat(
                F.smooth_l1_loss(self.pred_motion[..., :148], self.motions[..., :148], reduction='none'),
                mask=mask.unsqueeze(-1)
            )
        else:
            loss_vel = mean_flat(
                F.smooth_l1_loss(self.pred_motion[..., 4:67], self.motions[..., 4:67], reduction='none'),
                mask=mask.unsqueeze(-1)
            )

        loss = loss_rec + \
            self.cfg.training.lambda_global * loss_global + \
            self.cfg.training.lambda_fk * loss_fk + \
                self.cfg.training.lambda_commit * loss_commit +\
                self.cfg.training.lambda_expicit * loss_vel

        # return loss, loss_rec, loss_vel, loss_commit, perplexity
        # return loss, loss_rec, loss_percept, loss_commit, perplexity
        loss_logs = OrderedDict()
        loss_logs["loss"] = loss.item()
        loss_logs["loss_rec"] = loss_rec.item()
        loss_logs["loss_global"] = loss_global.item()
        loss_logs["loss_commit"] = loss_commit.item()
        loss_logs["loss_fk"] = loss_fk.item()
        loss_logs["loss_vel"] = loss_vel.item()
        loss_logs["perplexity"] = perplexity.item()
        return loss, loss_logs


    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }

        if self.cfg.training.ema:
            state["ema_model"] = self.ema_model.state_dict()
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device, weights_only=True)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        try:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            pass

        if self.cfg.training.ema:
            self.ema_model.load_state_dict(checkpoint['ema_model'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None, fk_func=None):
        self.vq_model.to(self.device)

        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.cfg.training.lr, betas=(0.9, 0.99), weight_decay=self.cfg.training.weight_decay)
        

        epoch = 0
        it = 0

        if self.cfg.training.ema:
            self.update_ema(self.ema_model, self.vq_model, decay=0)

        if self.cfg.exp.is_continue:
            model_dir = pjoin(self.cfg.exp.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_vq_model, 
        #                                                     T_max=self.cfg.training.max_epoch, 
        #                                                     eta_min=1e-6,
        #                                                     last_epoch=-1 if epoch==0 else epoch)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, 
                                                              milestones=self.cfg.training.milestones, 
                                                              gamma=self.cfg.training.gamma,
                                                              last_epoch=-1 if epoch==0 else epoch)


        # sys.exit()
        if self.cfg.data.name == 'snapmogen':
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe = evaluation_vqvae(
                self.cfg.exp.model_dir, eval_val_loader, 
                self.ema_model if self.cfg.training.ema else self.vq_model,
                self.logger, epoch, best_fid=1000,
                best_div=100, best_top1=0,
                best_top2=0, best_top3=0, best_matching=0, best_mpjpe=100, nfeats=self.cfg.data.dim_pose,
                eval_wrapper=eval_wrapper, device=self.device, fk_func=fk_func, save_ckpt=False, draw=True)
        else:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _ = evaluation_vqvae_hml(
            self.cfg.exp.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=1000,
            best_div=100, best_top1=0,
            best_top2=0, best_top3=0, best_matching=100,
            eval_wrapper=eval_wrapper, save=False)


        start_time = time.time()
        total_iters = self.cfg.training.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.cfg.training.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        # val_loss = 0
        # min_val_loss = np.inf
        # min_val_epoch = epoch
        # current_lr = self.cfg.training.lr
        def def_value():
            return 0.0
        logs = defaultdict(def_value, OrderedDict())

        while epoch < self.cfg.training.max_epoch:
            self.vq_model.train()
            for i, batch_data in enumerate(train_loader):
                # break
                it += 1
                if it < self.cfg.training.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.cfg.training.warm_up_iter, self.cfg.training.lr)
                # if self.cfg.model.use_attn:
                loss, loss_log = self.forward_attn(batch_data, self.vq_model, fk_func)
                # else:
                #     loss, loss_log = self.forward(batch_data, self.vq_model, fk_func)
                self.opt_vq_model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vq_model.parameters(), max_norm=0.5)
                self.opt_vq_model.step()

                if self.cfg.training.ema:
                    self.update_ema(self.ema_model, self.vq_model)
                
                for key, val in loss_log.items():
                    logs[key] += val

                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if it % self.cfg.training.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.cfg.training.log_every, it)
                        mean_loss[tag] = value / self.cfg.training.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                # if it % self.cfg.training.save_latest == 0:
                #     self.save(pjoin(self.cfg.exp.model_dir, 'latest.tar'), epoch, it)

            if it >= self.cfg.training.warm_up_iter:
                self.scheduler.step()
            self.save(pjoin(self.cfg.exp.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            # if epoch % self.cfg.save_every_e == 0:
            #     self.save(pjoin(self.cfg.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            # if epoch % self.cfg.training.eval_every_e != 0:
            #     continue

            print('Validation time:')
            self.vq_model.eval()
            
            val_logs = defaultdict(def_value, OrderedDict())
            eval_model = self.ema_model if self.cfg.training.ema else self.vq_model
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    # if self.cfg.model.use_attn:
                    #     loss, loss_log = self.forward_attn(batch_data, eval_model, fk_func)
                    # else:
                    loss, loss_log = self.forward_attn(batch_data, eval_model, fk_func)
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()
                    for key, val in loss_log.items():
                        val_logs[key] += val
            mean_loss = OrderedDict()
            for tag, value in val_logs.items():
                self.logger.add_scalar('Val/%s'%tag, value / len(val_loader), epoch)
                mean_loss[tag] = value / len(val_loader)
            
            print_val_loss(mean_loss, epoch)

            if self.cfg.data.name == 'snapmogen':
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe = evaluation_vqvae(
                self.cfg.exp.model_dir, eval_val_loader, 
                eval_model,
                self.logger, epoch, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, nfeats=self.cfg.data.dim_pose,
                best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, best_mpjpe=best_mpjpe,
                eval_wrapper=eval_wrapper, device=self.device, fk_func=fk_func, save_ckpt=True)
            else:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _ = evaluation_vqvae_hml(
                self.cfg.exp.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1,
                best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, eval_wrapper=eval_wrapper)


            if epoch % self.cfg.training.eval_every_e == 0:
                data = torch.cat([self.motions[:4], self.pred_motion[:4]], dim=0)
                # np.save(pjoin(self.cfg.eval_dir, 'E%04d.npy' % (epoch)), data)
                save_dir = pjoin(self.cfg.exp.eval_dir, 'E%04d' % (epoch))
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(data, save_dir)



class HVQTokenizerTrainer(VQTokenizerTrainer):
    def forward(self, batch_data):
        motions = batch_data.detach().to(self.device).float()
        pred_motion, loss_commit_b, loss_commit_t, perplexity_b, perplexity_t = self.vq_model(motions)

        self.motions = motions
        self.pred_motion = pred_motion

        loss_rec = self.l1_criterion(pred_motion, motions)
        pred_local_pos = pred_motion[..., 4 : self.cfg.data.joint_num * 3 + 4]
        local_pos = motions[..., 4 : self.cfg.data.joint_num * 3 + 4]
        loss_explicit = self.l1_criterion(pred_local_pos, local_pos)

        loss = (
            loss_rec
            + self.cfg.training.lambda_explict * loss_explicit
            + self.cfg.training.lambda_commit * (loss_commit_b + loss_commit_t)
        )

        # return loss, loss_rec, loss_vel, loss_commit, perplexity
        # return loss, loss_rec, loss_percept, loss_commit, perplexity
        return loss, loss_rec, loss_explicit, loss_commit_b, loss_commit_t, perplexity_b, perplexity_t
    

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None, fk_func=None):
        self.vq_model.to(self.device)

        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.cfg.training.lr, betas=(0.9, 0.99), weight_decay=self.cfg.training.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.cfg.training.milestones, gamma=self.cfg.training.gamma)

        epoch = 0
        it = 0
        if self.cfg.exp.is_continue:
            model_dir = pjoin(self.cfg.exp.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.cfg.training.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.cfg.training.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(eval_val_loader)))
        # val_loss = 0
        # min_val_loss = np.inf
        # min_val_epoch = epoch
        # current_lr = self.cfg.training.lr
        def def_value():
            return 0.0
        logs = defaultdict(def_value, OrderedDict())

        # sys.exit()
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe = evaluation_vqvae(
            self.cfg.exp.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=1000,
            best_div=100, best_top1=0,
            best_top2=0, best_top3=0, best_matching=0, best_mpjpe=100,
            eval_wrapper=eval_wrapper, device=self.device, fk_func=fk_func, save_ckpt=True)

        while epoch < self.cfg.training.max_epoch:
            self.vq_model.train()
            for i, batch_data in enumerate(train_loader):

                it += 1
                if it < self.cfg.training.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.cfg.training.warm_up_iter, self.cfg.training.lr)
                loss, loss_rec, loss_vel, loss_commit_b, loss_commit_t, perplexity_b, perplexity_t = self.forward(batch_data)
                self.opt_vq_model.zero_grad()
                loss.backward()
                self.opt_vq_model.step()

                if it >= self.cfg.training.warm_up_iter:
                    self.scheduler.step()
                
                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                # Note it not necessarily velocity, too lazy to change the name now
                logs['loss_vel'] += loss_vel.item()
                logs['loss_commit_b'] += loss_commit_b.item()
                logs['perplexity_b'] += perplexity_b.item()
                logs['loss_commit_t'] += loss_commit_t.item()
                logs['perplexity_t'] += perplexity_t.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if it % self.cfg.training.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.cfg.training.log_every, it)
                        mean_loss[tag] = value / self.cfg.training.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.cfg.training.save_latest == 0:
                    self.save(pjoin(self.cfg.exp.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.cfg.exp.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            # if epoch % self.cfg.save_every_e == 0:
            #     self.save(pjoin(self.cfg.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')
            self.vq_model.eval()
            val_loss_rec = []
            val_loss_vel = []
            val_loss_commit_b = []
            val_loss_commit_t = []
            val_loss = []
            val_perpexity_b = []
            val_perpexity_t = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, loss_rec, loss_vel, loss_commit_b, loss_commit_t, perplexity_b, perplexity_t = self.forward(batch_data)
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()
                    val_loss.append(loss.item())
                    val_loss_rec.append(loss_rec.item())
                    val_loss_vel.append(loss_vel.item())
                    val_loss_commit_b.append(loss_commit_b.item())
                    val_perpexity_b.append(perplexity_b.item())
                    val_loss_commit_t.append(loss_commit_t.item())
                    val_perpexity_t.append(perplexity_t.item())

            # val_loss = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)
            self.logger.add_scalar('Val/loss', sum(val_loss) / len(val_loss), epoch)
            self.logger.add_scalar('Val/loss_rec', sum(val_loss_rec) / len(val_loss_rec), epoch)
            self.logger.add_scalar('Val/loss_vel', sum(val_loss_vel) / len(val_loss_vel), epoch)
            self.logger.add_scalar('Val/loss_commit_b', sum(val_loss_commit_b) / len(val_loss), epoch)
            self.logger.add_scalar('Val/loss_perplexity_b', sum(val_perpexity_b) / len(val_loss_rec), epoch)
            self.logger.add_scalar('Val/loss_commit_t', sum(val_loss_commit_t) / len(val_loss), epoch)
            self.logger.add_scalar('Val/loss_perplexity_t', sum(val_perpexity_t) / len(val_loss_rec), epoch)

            print('Validation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit_t: %.5f, Commit_b: %.5f' %
                  (sum(val_loss)/len(val_loss), sum(val_loss_rec)/len(val_loss), 
                   sum(val_loss_vel)/len(val_loss), sum(val_loss_commit_t)/len(val_loss), 
                   sum(val_loss_commit_b)/len(val_loss)))

            # if sum(val_loss) / len(val_loss) < min_val_loss:
            #     min_val_loss = sum(val_loss) / len(val_loss)
            # # if sum(val_loss_vel) / len(val_loss_vel) < min_val_loss:
            # #     min_val_loss = sum(val_loss_vel) / len(val_loss_vel)
            #     min_val_epoch = epoch
            #     self.save(pjoin(self.cfg.model_dir, 'finest.tar'), epoch, it)
            #     print('Best Validation Model So Far!~')

            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe = evaluation_vqvae(
            self.cfg.exp.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=best_fid,
            best_div=best_div, best_top1=best_top1,
            best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, best_mpjpe=best_mpjpe,
            eval_wrapper=eval_wrapper, device=self.device, fk_func=fk_func, save_ckpt=True)


            if epoch % self.cfg.training.eval_every_e == 0:
                data = torch.cat([self.motions[:4], self.pred_motion[:4]], dim=0)
                # np.save(pjoin(self.cfg.eval_dir, 'E%04d.npy' % (epoch)), data)
                save_dir = pjoin(self.cfg.exp.eval_dir, 'E%04d' % (epoch))
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(data, save_dir)