import torch
from collections import defaultdict
import torch.optim as optim
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter # type: ignore
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_mask_transformer, evaluation_mask_transformer_hml
from model.transformer.tools import *

from einops import rearrange, repeat

from trainers.base_trainer import BaseTrainer

def def_value():
    return 0.0

class MaskTransformerTrainer(BaseTrainer):
    def __init__(self, cfg, t2m_transformer, vq_model, device):
        self.cfg = cfg
        self.t2m_transformer = t2m_transformer
        self.vq_model = vq_model
        self.device = device
        self.vq_model.eval()

        if cfg.exp.is_train:
            self.logger = SummaryWriter(cfg.exp.log_dir)


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q)
        code_idx, _ = self.vq_model.encode(motion[..., :self.cfg.data.dim_pose], m_lens)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        # loss_dict = {}
        # self.pred_ids = []
        # self.acc = []

        _loss, _pred_ids, _acc = self.t2m_transformer(code_idx, conds, m_lens)

        return _loss, _acc

    def update(self, batch_data, step_scheduler=True):
        loss, acc = self.forward(batch_data)

        self.opt_t2m_transformer.zero_grad()
        loss.backward()
        self.opt_t2m_transformer.step()
        if step_scheduler:
            self.scheduler.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.t2m_transformer.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device, weights_only=True)
        missing_keys, unexpected_keys = self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.t2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_t2m_transformer = optim.AdamW(self.t2m_transformer.parameters(), betas=(0.9, 0.99), lr=self.cfg.training.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
                                                        milestones=self.cfg.training.milestones,
                                                        gamma=self.cfg.training.gamma)
        

        epoch = 0
        it = 0

        if self.cfg.exp.is_continue:
            model_dir = pjoin(self.cfg.exp.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.cfg.training.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.cfg.training.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt_t2m_transformer,
        #                                                                 T_0=50_000, 
        #                                                                 T_mult=2, 
        #                                                                 eta_min=0, 
        #                                                                 last_epoch=-1 if it==0 else it)

        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt_t2m_transformer,
        #                                                     T_max=self.cfg.training.max_epoch,
        #                                                     eta_min=0,
        #                                                     last_epoch=-1 if epoch==0 else epoch)  

        if self.cfg.data.name == 'snapmogen':
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching = evaluation_mask_transformer(
                self.cfg.exp.model_dir, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch,
                best_fid=1000, best_div=100,
                best_top1=0, best_top2=0, best_top3=0,
                best_matching=0, eval_wrapper=eval_wrapper, device=self.device,
                plot_func=plot_eval, save_ckpt=False, save_anim=False
            )
        else:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching = evaluation_mask_transformer_hml(
                self.cfg.exp.model_dir, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch,
                best_fid=1000, best_div=100,
                best_top1=0, best_top2=0, best_top3=0,
                best_matching=100, eval_wrapper=eval_wrapper, device=self.device,
                plot_func=plot_eval, save_ckpt=False, save_anim=False
            )
        best_acc = 0.

        while epoch < self.cfg.training.max_epoch:
            self.t2m_transformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.cfg.training.warm_up_iter:
                    self.update_lr_warm_up(it, self.cfg.training.warm_up_iter, self.cfg.training.lr)

                loss, acc = self.update(batch, it >= self.cfg.training.warm_up_iter)
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

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

            # if it > self.cfg.training.warm_up_iter:
            #     self.scheduler.step()
            self.save(pjoin(self.cfg.exp.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            print('Validation time:')
            self.vq_model.eval()
            self.t2m_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

            self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_acc) > best_acc:
                print(f"Improved accuracy from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                self.save(pjoin(self.cfg.exp.model_dir, 'net_best_acc.tar'), epoch, it)
                best_acc = np.mean(val_acc)
            # if epoch%self.cfg.training.eval_every_e==0:
            if self.cfg.data.name == 'snapmogen':
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching = evaluation_mask_transformer(
                    self.cfg.exp.model_dir, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, eval_wrapper=eval_wrapper, device=self.device,
                    plot_func=plot_eval, save_ckpt=True, save_anim=False
                )
            else:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching = evaluation_mask_transformer_hml(
                    self.cfg.exp.model_dir, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, eval_wrapper=eval_wrapper, device=self.device,
                    plot_func=plot_eval, save_ckpt=True, save_anim=False
                )