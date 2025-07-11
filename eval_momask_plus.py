import warnings
warnings.filterwarnings("ignore")

import os
from os.path import join as pjoin

import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.utils.data import DataLoader

from model.vq.rvq_model import HRVQVAE
from model.evaluator.evaluator_wrapper import EvaluatorWrapper
from dataset.dataset import TextMotionDataset
from model.transformer.transformer import MoMaskPlus
from config.load_config import load_config

import utils.eval_t2m as eval_t2m
from utils.fixseeds import fixseed
import collections

import numpy as np

def load_vq_model(vq_cfg, device):
    # print(cfg.exp)
    # vq_cfg = load_config(pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'vq', cfg.vq_name, 'residual_vqvae.yaml'))

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

    ckpt = torch.load(pjoin(vq_cfg.exp.root_ckpt_dir, vq_cfg.data.name, 'vq', vq_cfg.exp.name, 'model',mask_trans_cfg.vq_ckpt),
                            map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_cfg.exp.name} from epoch {ckpt["ep"]}')
    vq_model.to(device)
    vq_model.eval()
    return vq_model

def load_trans_model(t2m_cfg, which_model, device):
    t2m_transformer = MoMaskPlus(
        code_dim=t2m_cfg.vq.code_dim,
        latent_dim=t2m_cfg.model.latent_dim,
        ff_size=t2m_cfg.model.ff_size,
        num_layers=t2m_cfg.model.n_layers,
        num_heads=t2m_cfg.model.n_heads,
        dropout=t2m_cfg.model.dropout,
        text_dim=t2m_cfg.text_embedder.dim_embed,
        cond_drop_prob=t2m_cfg.training.cond_drop_prob,
        device=device,
        cfg=t2m_cfg,
        full_length=t2m_cfg.data.max_motion_length//4,
        scales=vq_cfg.quantizer.scales
    )
    ckpt = torch.load(pjoin(t2m_cfg.exp.root_ckpt_dir, t2m_cfg.data.name, "momask_plus", t2m_cfg.exp.name, 'model', which_model),
                      map_location=cfg.device, weights_only=True)
    t2m_transformer.load_state_dict(ckpt["t2m_transformer"])

    t2m_transformer.to(device)
    t2m_transformer.eval()
    print(f'Loading Mask Transformer {t2m_cfg.exp.name} from epoch {ckpt["ep"]}!')
    print(f'Loading Mask Transformer {t2m_cfg.exp.name} from epoch {ckpt["ep"]}!', file=f, flush=True)
    return t2m_transformer


if __name__ == '__main__':
    # parser = EvalT2MOptions()
    # opt = parser.parse()
    cfg = load_config("./config/eval_momaskplus.yaml")
    fixseed(cfg.seed)

    if cfg.device != 'cpu':
        torch.cuda.set_device(cfg.device)
    device = torch.device(cfg.device)
    torch.autograd.set_detect_anomaly(True)

    cfg.checkpoint_dir = pjoin(cfg.root_ckpt_dir, cfg.data.name, 'momask_plus', cfg.mask_trans_name)
    cfg.model_dir = pjoin(cfg.checkpoint_dir, 'model')
    cfg.eval_dir = pjoin(cfg.checkpoint_dir, 'eval')
    # cfg.log_dir = pjoin(cfg.root_log_dir, cfg.data.name, 'momask_plus',cfg.mask_trans_name)

    os.makedirs(cfg.eval_dir, exist_ok=True)

    out_path = pjoin(cfg.eval_dir, "%s.log"%cfg.ext)

    f = open(pjoin(out_path), 'w')

    mask_trans_cfg = load_config(pjoin(cfg.root_ckpt_dir, cfg.data.name, 'momask_plus', cfg.mask_trans_name, 'train_momaskplus.yaml'))

    vq_cfg = load_config(pjoin(cfg.root_ckpt_dir, cfg.data.name, 'vq', mask_trans_cfg.vq_name, 'residual_vqvae.yaml'))
    mask_trans_cfg.vq = vq_cfg.quantizer
    # res_trans_cfg.vq = vq_cfg.quantizer

    vq_model = load_vq_model(vq_cfg, device)

    cfg.data.feat_dir = pjoin(cfg.data.root_dir, 'renamed_feats')
    meta_dir = pjoin(cfg.data.root_dir, 'meta_data')
    data_split_dir = pjoin(cfg.data.root_dir, 'data_split_info')
    all_caption_path = pjoin(cfg.data.root_dir, 'all_caption_clean.json')

    test_mid_split_file = pjoin(data_split_dir, 'test_fnames.txt')
    test_cid_split_file = pjoin(data_split_dir, 'test_ids.txt')

    mean = np.load(pjoin(meta_dir, 'mean.npy'))
    std = np.load(pjoin(meta_dir, 'std.npy'))
    eval_dataset = TextMotionDataset(cfg, mean, std, test_mid_split_file, test_cid_split_file, all_caption_path)
    eval_cfg = load_config(pjoin('checkpoint_dir/snapmogen/evaluator/eval_klde-5_late-5_nlayer6_norm/evaluator.yaml'))
    eval_wrapper = EvaluatorWrapper(eval_cfg, device=device)

    # eval_cfg = load_config(pjoin('checkpoint_dir/snapmogen/evaluator/evalv2_rec1_cst0.1_ld256/evaluator_v2.yaml'))
    # eval_wrapper = EvaluatorWrapperV2(eval_cfg, device=device)

    eval_loader = DataLoader(eval_dataset, batch_size=cfg.matching_pool_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=True)

    # model_dir = pjoin(cfg.)
    for file in os.listdir(cfg.model_dir):
        if cfg.which_epoch != "all" and cfg.which_epoch not in file:
            continue
        print('loading checkpoint {}'.format(file))
        t2m_transformer = load_trans_model(mask_trans_cfg, file, device)

        # repeat_time = 20
        for cs in cfg.cond_scales:
            for ts in cfg.time_steps:
                fid = []
                div = []
                top1 = []
                top2 = []
                top3 = []
                matching = []
                mm = []
                for i in range(cfg.repeat_time):
                    
                    print(f'Guidance scale: {cs}, time step: {ts}')
                    print(f'Guidance scale: {cs}, time step: {ts}', file=f, flush=True)

                    with torch.no_grad():
                        best_fid, best_div, Rprecision, best_matching, best_mm = (
                            eval_t2m.evaluation_momask_plus(
                                eval_loader,
                                vq_model,
                                t2m_transformer,
                                i,
                                eval_wrapper=eval_wrapper,
                                time_steps=ts,
                                cond_scale=cs,
                                temperature=cfg.temperature,
                                gsample=cfg.gsample,
                                topkr=cfg.topkr,
                                cal_mm=cfg.cal_mm,
                            )
                        )
                    fid.append(best_fid)
                    div.append(best_div)
                    top1.append(Rprecision[0])
                    top2.append(Rprecision[1])
                    top3.append(Rprecision[2])
                    matching.append(best_matching)
                    mm.append(best_mm)

                fid = np.array(fid)
                div = np.array(div)
                top1 = np.array(top1)
                top2 = np.array(top2)
                top3 = np.array(top3)
                matching = np.array(matching)
                mm = np.array(mm)

                print(f'{file} final result (Guidance scale: {cs}, time step: {ts}):')
                print(f'{file} final result Guidance scale: {cs}, time step: {ts}():', file=f, flush=True)

                msg_final = (
                    f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(cfg.repeat_time):.3f}\n"
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(cfg.repeat_time):.3f}\n"
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(cfg.repeat_time):.3f}, "
                    f"TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(cfg.repeat_time):.3f}, "
                    f"TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(cfg.repeat_time):.3f}\n"
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(cfg.repeat_time):.3f}\n"
                    f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(cfg.repeat_time):.3f}\n\n"
                )
                # logger.info(msg_final)
                print(msg_final)
                print(msg_final, file=f, flush=True)

    f.close()


# python eval_t2m_trans.py --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_vq --dataset_name t2m --gpu_id 3 --cond_scale 4 --time_steps 18 --temperature 1 --topkr 0.9 --gumbel_sample --ext cs4_ts18_tau1_topkr0.9_gs
