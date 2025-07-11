import os
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader

from model.vq.rvq_model import HRVQVAE
from model.evaluator.evaluator_wrapper import EvaluatorWrapper
from trainers.ae_trainer import VQTokenizerTrainer
from config.load_config import load_config

from dataset.dataset import  TextMotionDataset
from utils.paramUtil import kinematic_chain
from utils import bvh_io
from utils.utils import plot_3d_motion
from common.skeleton import Skeleton
from utils.motion_process_bvh import recover_pos_from_rot
import numpy as np
from utils.fixseeds import *

import shutil

def forward_kinematic_func(data):
    motions = train_dataset.inv_transform(data)
    global_pos = recover_pos_from_rot(motions, 
                                      joints_num=cfg.data.joint_num, 
                                      skeleton=skeleton)
    return global_pos

def plot_t2m(data, save_dir):
    global_pos = forward_kinematic_func(data).detach().cpu().numpy()
    # data = train_dataset.inv_transform(data)
    for i in range(len(global_pos)):
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        plot_3d_motion(save_path, 
                       kinematic_chain, 
                       global_pos[i], 
                       title="None", 
                       fps=30, 
                       radius=100)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    cfg = load_config('config/residual_vqvae.yaml')
    cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'vq', cfg.exp.name)

    if cfg.exp.is_continue:
        n_cfg = load_config(pjoin(cfg.exp.checkpoint_dir, 'residual_vqvae.yaml'))
        n_cfg.exp.is_continue = True
        n_cfg.exp.device = cfg.exp.device
        n_cfg.exp.checkpoint_dir = cfg.exp.checkpoint_dir
        cfg = n_cfg
        # print(cfg)
    else:
        os.makedirs(cfg.exp.checkpoint_dir, exist_ok=True)
        shutil.copy('config/residual_vqvae.yaml', cfg.exp.checkpoint_dir)
    
    fixseed(cfg.exp.seed)

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

    cfg.data.feat_dir = pjoin(cfg.data.root_dir, 'renamed_feats')
    meta_dir = pjoin(cfg.data.root_dir, 'meta_data')
    data_split_dir = pjoin(cfg.data.root_dir, 'data_split_info')
    all_caption_path = pjoin(cfg.data.root_dir, 'all_caption_clean.json')

    train_mid_split_file = pjoin(data_split_dir, 'train_fnames.txt')
    train_cid_split_file = pjoin(data_split_dir, 'train_ids.txt')

    val_mid_split_file = pjoin(data_split_dir, 'val_fnames.txt')
    val_cid_split_file = pjoin(data_split_dir, 'val_ids.txt')

    template_anim = bvh_io.load(pjoin(cfg.data.root_dir, 'renamed_bvhs', 'm_ep2_00086.bvh'))
    skeleton = Skeleton(template_anim.offsets, template_anim.parents, device=device)
    

    mean = np.load(pjoin(meta_dir, 'mean.npy'))
    std = np.load(pjoin(meta_dir, 'std.npy'))

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

    # if cfg.model.use_attn:
    train_dataset = TextMotionDataset(cfg, mean, std, train_mid_split_file, train_cid_split_file, all_caption_path)
    val_dataset = TextMotionDataset(cfg, mean, std, val_mid_split_file, val_cid_split_file, all_caption_path)
    # else:
    #     train_dataset = MotionDataset(cfg, mean, std, train_mid_split_file, train_cid_split_file)
    #     val_dataset = MotionDataset(cfg, mean, std, val_mid_split_file, val_cid_split_file)
    eval_dataset = TextMotionDataset(cfg, mean, std, val_mid_split_file, val_cid_split_file, all_caption_path)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=True)
    
    eval_cfg = load_config(pjoin('checkpoint_dir/snapmogen/evaluator/eval_klde-5_late-5_nlayer6_norm/evaluator.yaml'))
    eval_wrapper = EvaluatorWrapper(eval_cfg, device=device)

    eval_loader = DataLoader(eval_dataset, batch_size=eval_cfg.matching_pool_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=True)
    
    trainer.train(train_loader, val_loader, eval_loader, eval_wrapper, plot_t2m, forward_kinematic_func)

