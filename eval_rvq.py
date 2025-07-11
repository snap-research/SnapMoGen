import os
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader

from model.vq.rvq_model import RVQVAE, HRVQVAE
from model.vq.hvq_model import HVQVAE
from model.evaluator.evaluator_wrapper import EvaluatorWrapper
from trainers.ae_trainer import VQTokenizerTrainer
from config.load_config import load_config

from dataset.dataset import MotionDataset, TextMotionDataset
from utils.paramUtil import kinematic_chain
from utils import bvh_io
from utils.utils import plot_3d_motion
from common.skeleton import Skeleton
from utils.motion_process_bvh import recover_pos_from_rot
import numpy as np
from utils.fixseeds import *
from utils.eval_t2m import evaluation_vqvae

import shutil

def forward_kinematic_func(data):
    motions = eval_dataset.inv_transform(data)
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


def load_vq_model(cfg, device):
    # print(cfg.exp)
    # vq_cfg = load_config(pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'vq', cfg.vq_name, 'residual_vqvae.yaml'))
    vq_model = HRVQVAE(cfg,
            cfg.data.dim_pose,
            cfg.model.down_t,
            cfg.model.stride_t,
            cfg.model.width,
            cfg.model.depth,
            cfg.model.dilation_growth_rate,
            cfg.model.vq_act,
            cfg.model.use_attn,
            cfg.model.vq_norm)
        

    ckpt = torch.load(pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'vq', cfg.exp.name, 'model', 'net_best_fid.tar'),
                            map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    # model_key = "ema_model"
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {cfg.exp.name} from epoch {ckpt["ep"]} and iteration {ckpt["total_it"] if "total_it" in ckpt else 0}')
    vq_model.to(device)
    vq_model.eval()
    return vq_model

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    cfg = load_config('config/residual_vqvae.yaml')
    cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'vq', cfg.exp.name)

    # if cfg.exp.is_continue:
    n_cfg = load_config(pjoin(cfg.exp.checkpoint_dir, 'residual_vqvae.yaml'))
    # n_cfg.exp.is_continue = True
    n_cfg.exp.device = cfg.exp.device
    n_cfg.exp.checkpoint_dir = cfg.exp.checkpoint_dir
    cfg = n_cfg
        # print(cfg)
    
    fixseed(cfg.exp.seed)

    if cfg.exp.device != 'cpu':
        torch.cuda.set_device(cfg.exp.device)

    torch.autograd.set_detect_anomaly(True)
    
    device = torch.device(cfg.exp.device)

    cfg.exp.model_dir = pjoin(cfg.exp.checkpoint_dir, 'model')
    cfg.exp.eval_dir = pjoin(cfg.exp.checkpoint_dir, 'animation')
    cfg.exp.log_dir = pjoin(cfg.exp.root_log_dir, cfg.data.name, 'vq',cfg.exp.name)


    cfg.data.feat_dir = pjoin(cfg.data.root_dir, 'renamed_feats')
    meta_dir = pjoin(cfg.data.root_dir, 'meta_data')
    data_split_dir = pjoin(cfg.data.root_dir, 'data_split_info1')
    all_caption_path = pjoin(cfg.data.root_dir, 'all_caption_clean.json')


    val_mid_split_file = pjoin(data_split_dir, 'test_fnames.txt')
    val_cid_split_file = pjoin(data_split_dir, 'test_ids.txt')

    template_anim = bvh_io.load(pjoin(cfg.data.root_dir, 'renamed_bvhs', 'm_ep2_00086.bvh'))
    skeleton = Skeleton(template_anim.offsets, template_anim.parents, device=device)
    

    mean = np.load(pjoin(meta_dir, 'mean.npy'))
    std = np.load(pjoin(meta_dir, 'std.npy'))


    net = load_vq_model(cfg, device)

    eval_dataset = TextMotionDataset(cfg, mean, std, val_mid_split_file, val_cid_split_file, all_caption_path)
    
    eval_cfg = load_config(pjoin('checkpoint_dir/snapmogen/evaluator/eval_klde-5_late-5_nlayer6_norm/evaluator.yaml'))
    eval_wrapper = EvaluatorWrapper(eval_cfg, device=device)

    eval_loader = DataLoader(eval_dataset, batch_size=eval_cfg.matching_pool_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=True)
    
    # trainer.train(None, val_loader, eval_loader, eval_wrapper, plot_t2m, forward_kinematic_func)
    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe = evaluation_vqvae(
            cfg.exp.model_dir, eval_loader, net, None, 0, best_fid=1000,
            best_div=100, best_top1=0,
            best_top2=0, best_top3=0, best_matching=0, best_mpjpe=100, nfeats=cfg.data.dim_pose,
            eval_wrapper=eval_wrapper, device=device, fk_func=forward_kinematic_func, save_ckpt=False, draw=False)

