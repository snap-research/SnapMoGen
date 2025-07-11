import os
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader

from model.vq.rvq_model import HRVQVAE
from model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper
from model.evaluator.hml.dataset_motion_loader import get_dataset_motion_loader
from model.transformer.transformer import MoMaskPlus
from trainers.transformer_trainer import MaskTransformerTrainer
from config.load_config import load_config

from dataset.humanml3d_dataset import MotionDataset, Text2MotionDataset
from utils.get_opt import get_opt

from utils.paramUtil import t2m_kinematic_chain
from utils import bvh_io
from utils.utils import plot_3d_motion
from common.skeleton import Skeleton
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


def load_vq_model(cfg, device):
    # print(cfg.exp)
    vq_cfg = load_config(pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'vq', cfg.vq_name, 'residual_vqvae_hml.yaml'))

    vq_model = HRVQVAE(vq_cfg,
            vq_cfg.data.dim_pose,
            vq_cfg.model.down_t,
            vq_cfg.model.stride_t,
            vq_cfg.model.width,
            vq_cfg.model.depth,
            vq_cfg.model.dilation_growth_rate,
            vq_cfg.model.vq_act,
            vq_cfg.model.use_attn,
            vq_cfg.model.vq_norm)
        
    ckpt = torch.load(pjoin(vq_cfg.exp.root_ckpt_dir, vq_cfg.data.name, 'vq', vq_cfg.exp.name, 'model',cfg.vq_ckpt),
                            map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_cfg.exp.name} from epoch {ckpt["ep"]}')
    vq_model.to(device)
    vq_model.eval()
    return vq_model, vq_cfg



if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    cfg = load_config('config/train_momaskplus_hml.yaml')
    cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'momask_plus', cfg.exp.name)

    if cfg.exp.is_continue:
        n_cfg = load_config(pjoin(cfg.exp.checkpoint_dir, 'train_momaskplus_hml.yaml'))
        n_cfg.exp.is_continue = True
        n_cfg.exp.device = cfg.exp.device
        n_cfg.exp.checkpoint_dir = cfg.exp.checkpoint_dir
        cfg = n_cfg
        # print(cfg)
    else:
        os.makedirs(cfg.exp.checkpoint_dir, exist_ok=True)
        shutil.copy('config/train_momaskplus_hml.yaml', cfg.exp.checkpoint_dir)

    fixseed(cfg.exp.seed)

    if cfg.exp.device != 'cpu':
        torch.cuda.set_device(cfg.exp.device)

    torch.autograd.set_detect_anomaly(True)

    device = torch.device(cfg.exp.device)

    cfg.exp.model_dir = pjoin(cfg.exp.checkpoint_dir, 'model')
    cfg.exp.eval_dir = pjoin(cfg.exp.checkpoint_dir, 'animation')
    cfg.exp.log_dir = pjoin(cfg.exp.root_log_dir, cfg.data.name, 'momask_plus',cfg.exp.name)

    os.makedirs(cfg.exp.model_dir, exist_ok=True)
    os.makedirs(cfg.exp.eval_dir, exist_ok=True)
    os.makedirs(cfg.exp.log_dir, exist_ok=True)

    dataset_opt_path = 'checkpoint_dir/humanml3d/Comp_v6_KLD005/opt.txt'


    cfg.exp.model_dir = pjoin(cfg.exp.checkpoint_dir, 'model')
    cfg.exp.eval_dir = pjoin(cfg.exp.checkpoint_dir, 'animation')
    cfg.exp.log_dir = pjoin(cfg.exp.root_log_dir, cfg.data.name, 'momask_plus',cfg.exp.name)

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
    
    vq_model, vq_cfg = load_vq_model(cfg, device=device)

    if 'rvq' in cfg.vq_name:
        cfg.vq = vq_cfg.quantizer
    elif 'hvq' in cfg.vq_name:
        cfg.vq = vq_cfg.quantizer
        cfg.vq.nb_code = vq_cfg.quantizer.nb_code_t
        cfg.vq.code_dim = vq_cfg.quantizer.code_dim_t
    
    # cfg.vq.nb_code = vq_cfg.quantizer.nb_code

    t2m_transformer = MoMaskPlus(
        code_dim=cfg.vq.code_dim,
        latent_dim=cfg.model.latent_dim,
        ff_size=cfg.model.ff_size,
        num_layers=cfg.model.n_layers,
        num_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        text_dim=cfg.text_embedder.dim_embed,
        cond_drop_prob=cfg.training.cond_drop_prob,
        device=device,
        cfg=cfg,
        full_length=cfg.data.max_motion_length//4,
        scales=vq_cfg.quantizer.scales
    )

    pc_vq = sum(param.numel() for param in t2m_transformer.parameters())
    print(t2m_transformer)
    # print("Total parameters of discriminator net: {}".format(pc_vq))
    # all_params += pc_vq_dis

    print('Total parameters of all models: {}M'.format(pc_vq/1000_000))
    print(device)

    trainer = MaskTransformerTrainer(cfg, t2m_transformer, vq_model=vq_model, device=device)


    train_dataset = Text2MotionDataset(wrapper_opt, mean, std, train_cid_split_file)
    eval_dataset = Text2MotionDataset(wrapper_opt, mean, std, val_cid_split_file)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(eval_dataset, batch_size=cfg.training.batch_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=True)
    
    eval_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=device)


    trainer.train(train_loader, val_loader, eval_loader, eval_wrapper, plot_t2m)
