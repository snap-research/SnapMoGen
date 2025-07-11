import os
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader

from model.vq.rvq_model import HRVQVAE
from model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper
from model.evaluator.hml.dataset_motion_loader import get_dataset_motion_loader
from trainers.ae_trainer import VQTokenizerTrainer
from config.load_config import load_config

from dataset.humanml3d_dataset import MotionDataset, Text2MotionDataset
from utils.get_opt import get_opt
from utils.paramUtil import t2m_kinematic_chain
from utils.utils import plot_3d_motion
from utils.motion_process_hml import recover_from_ric
import numpy as np
from utils.fixseeds import *

import shutil

def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data.cpu().detach().numpy())
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        plot_3d_motion(save_path, t2m_kinematic_chain, joint, title="None", fps=20, radius=4)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    cfg = load_config('config/residual_vqvae_hml.yaml')
    cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'vq', cfg.exp.name)

    if cfg.exp.is_continue:
        n_cfg = load_config(pjoin(cfg.exp.checkpoint_dir, 'residual_vqvae_hml.yaml'))
        n_cfg.exp.is_continue = True
        n_cfg.exp.device = cfg.exp.device
        n_cfg.exp.checkpoint_dir = cfg.exp.checkpoint_dir
        cfg = n_cfg
        # print(cfg)
    else:
        os.makedirs(cfg.exp.checkpoint_dir, exist_ok=True)
        shutil.copy('config/residual_vqvae_hml.yaml', cfg.exp.checkpoint_dir)
    
    fixseed(cfg.exp.seed)

    dataset_opt_path = 'checkpoint_dir/humanml3d/Comp_v6_KLD005/opt.txt'

    if cfg.exp.device != 'cpu':
        torch.cuda.set_device(cfg.exp.device)

    torch.autograd.set_detect_anomaly(True)
    
    device = torch.device(cfg.exp.device)

    cfg.exp.model_dir = pjoin(cfg.exp.checkpoint_dir, 'model')
    cfg.exp.eval_dir = pjoin(cfg.exp.checkpoint_dir, 'animation')
    cfg.exp.log_dir = pjoin(cfg.exp.root_log_dir, cfg.data.name, 'vq',cfg.exp.name)

    os.makedirs(cfg.exp.model_dir, exist_ok=True)
    os.makedirs(cfg.exp.eval_dir, exist_ok=True)
    os.makedirs(cfg.exp.log_dir, exist_ok=True)

    cfg.data.feat_dir = pjoin(cfg.data.root_dir, 'new_joint_vecs')

    train_cid_split_file = pjoin(cfg.data.root_dir, 'train.txt')
    val_cid_split_file = pjoin(cfg.data.root_dir, 'val.txt')
    

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))

    net = HRVQVAE(cfg,
                cfg.data.dim_pose,
                cfg.model.down_t,
                cfg.model.stride_t,
                cfg.model.width,
                cfg.model.depth,
                cfg.model.dilation_growth_rate,
                cfg.model.vq_act,
                cfg.model.use_attn,
                cfg.model.vq_norm
                )

    pc_vq = sum(param.numel() for param in net.parameters())
    print(net)
    # print("Total parameters of discriminator net: {}".format(pc_vq))
    # all_params += pc_vq_dis

    print('Total parameters of all models: {}M'.format(pc_vq/1000_000))
    print("Device: %s"%device)

    trainer = VQTokenizerTrainer(cfg, vq_model=net, device=device)

    train_dataset = Text2MotionDataset(wrapper_opt, mean, std, train_cid_split_file)
    eval_dataset = Text2MotionDataset(wrapper_opt, mean, std, val_cid_split_file)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(eval_dataset, batch_size=cfg.training.batch_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=True)
    
    eval_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=device)

    
    trainer.train(train_loader, val_loader, eval_loader, eval_wrapper, plot_t2m, None)

