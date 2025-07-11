import os

# import clip
import numpy as np
import torch
# from scipy import linalg
from utils.metrics import *
import torch.nn.functional as F
from tqdm import tqdm


def length_to_mask(length, max_len, device: torch.device = None) -> torch.Tensor: # type: ignore
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length)
    
    length = length.to(device)
    # max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask


@torch.no_grad()
def evaluation_evaluator(out_dir, eval_val_loader, writer, ep, best_top1, best_top2, best_top3, 
                         best_matching, eval_model, device, save_ckpt=True, draw=True):
    # eval_model.eval()

    def save(file_path, ep):
        state = {
            "latent_enc": eval_model.latent_enc.state_dict(),
            "text_enc": eval_model.text_enc.state_dict(),
            "ep": ep,
        }

        if "motion_enc" in eval_model.state_dict():
            state["motion_enc"] = eval_model.motion_enc.state_dict()
        
        # if "text_enc" in eval_model.state_dict():
        #     state["text_enc"] = eval_model.text_enc.state_dict(),


        torch.save(state, file_path)

    motion_annotation_list = []

    R_precision_real = 0

    nb_sample = 0
    matching_score_real = 0
    for batch in eval_val_loader:
        # print(len(batch))
        texts, motions, m_lengths = batch

        motions = motions[..., :148]
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        et, _ = eval_model.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_model.encode_motion(motions, m_lengths, sample_mean=True)

        bs, _ = motions.shape[0], motions.shape[1]


        motion_annotation_list.append(fid_em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample

    matching_score_real = matching_score_real / nb_sample

    msg = "--> \t Eva. Ep %d:, Diversity Real. %.4f, R_precision_real. (%.4f, %.4f, %.4f), matching_score_real. %.4f"%\
          (ep, diversity_real, R_precision_real[0],R_precision_real[1], R_precision_real[2], matching_score_real ) # type: ignore
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('Eval/Diversity', diversity_real, ep)
        writer.add_scalar('Eval/top1', R_precision_real[0], ep) # type: ignore
        writer.add_scalar('Eval/top2', R_precision_real[1], ep)
        writer.add_scalar('Eval/top3', R_precision_real[2], ep)
        writer.add_scalar('Eval/matching_score', matching_score_real, ep)


    # msg = "--> --> \t Diversity %.5f !!!"%(diversity_real)
    # print(msg)
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision_real[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision_real[0])
        if draw: print(msg)
        best_top1 = R_precision_real[0]
        if save_ckpt:
            save(os.path.join(out_dir, 'net_best_top1.tar'), ep)
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision_real[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision_real[1])
        if draw: print(msg)
        best_top2 = R_precision_real[1]

    if R_precision_real[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision_real[2])
        if draw: print(msg)
        best_top3 = R_precision_real[2]

    if matching_score_real > best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_real)
        if draw: print(msg)
        best_matching = matching_score_real
        if save_ckpt:
            # save(os.path.join(out_dir, 'net_best_mm.tar'),
            #      ep
            #      )
            save(os.path.join(out_dir, 'net_best_mm.tar'), ep)
    # eval_model.train()

    return diversity_real, best_top1, best_top2, best_top3, best_matching


@torch.no_grad()
def evaluation_vqvae(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_top1,
                     best_top2, best_top3, best_matching, best_mpjpe, nfeats,
                     eval_wrapper, device, fk_func, save_ckpt=True, draw=True):
    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0

    mpjpe_error_sum = 0
    frame_count_sum = 0

    net.eval()
    for batch in val_loader:
        texts, motions, m_lengths = batch

        # motions = motions[..., :148]
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        et, _ = eval_wrapper.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_wrapper.encode_motion(motions[..., :148], m_lengths, sample_mean=True)
        bs, _ = motions.shape[0], motions.shape[1]


        if 'vq' in out_dir:
            _, all_codes = net.encode(motions[...,:nfeats], m_lengths.clone())
        else:
            all_codes = net.encode(motions[..., :nfeats])
        rec_motions = net.decode(all_codes, m_lengths.clone())
        fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(rec_motions[..., :148], m_lengths, sample_mean=True)

        batch_mpjpe_error, batch_frame_count = calculate_mpjpe(
            fk_func(rec_motions), 
            fk_func(motions),
            mask=length_to_mask(m_lengths, motions.shape[1]),
            only_local=False
            )
        
        mpjpe_error_sum += batch_mpjpe_error
        frame_count_sum += batch_frame_count

        motion_pred_list.append(fid_em_pred)
        motion_annotation_list.append(fid_em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe_error = mpjpe_error_sum / frame_count_sum

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f, mpjpe. %.4f"%\
          (ep, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred, mpjpe_error )
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('Eval/FID', fid, ep)
        writer.add_scalar('Eval/Diversity', diversity, ep)
        writer.add_scalar('Eval/top1', R_precision[0], ep)
        writer.add_scalar('Eval/top2', R_precision[1], ep)
        writer.add_scalar('Eval/top3', R_precision[2], ep)
        writer.add_scalar('Eval/matching_score', matching_score_pred, ep)
        writer.add_scalar('Eval/mpjpe', mpjpe_error, ep)

    draw = True
    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save_ckpt:
            torch.save({'model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]
        # if save_ckpt:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred > best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save_ckpt:
            torch.save({'model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mm.tar'))

    if mpjpe_error < best_mpjpe:
        msg = f"--> --> \t mpjpe Improved from %.5f to %.5f !!!" % (best_mpjpe, mpjpe_error)
        if draw: print(msg)
        best_mpjpe = mpjpe_error
        if save_ckpt:
            torch.save({'model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mpjpe.tar'))

    # if save:
    #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    # net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe


@torch.no_grad()
def evaluation_vqvae_hml(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_top1,
                     best_top2, best_top3, best_matching, eval_wrapper, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        m_length = m_length.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        # pred_pose_eval, loss_commit, perplexity = net(motion)
        _, all_codes = net.encode(motion, m_length.clone())
        pred_pose_eval = net.decode(all_codes, m_length.clone())

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"%\
          (ep, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred )
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mm.tar'))

    # if save:
    #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_mask_transformer(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, device, plot_func, time_steps = 10,
                           cond_scale = 4, save_ckpt=False, save_anim=False, draw=True):


    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0


    nb_sample = 0
    # for i in range(1):
    for batch in tqdm(val_loader):
        texts, motions, m_lengths = batch
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        et, _ = eval_wrapper.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_wrapper.encode_motion(motions[..., :148], m_lengths, sample_mean=True)
        bs, _ = motions.shape[0], motions.shape[1]

        # mids, _ = vq_model.encode(motions)
        # mids = mids[..., 0:1]
        # motion_codes = motion_codes.permute(0, 2, 1)
        mids = trans.generate(texts, m_lengths//4, time_steps, cond_scale, temperature=1)
        pred_motions = vq_model.forward_decoder(mids, m_lengths.clone())


        # mids, _ = vq_model.encode(motions)
        # mids = mids['top']

        fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths, sample_mean=True)

        motion_annotation_list.append(fid_em)
        motion_pred_list.append(fid_em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    
    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    if draw: print(msg)

    if draw:
        writer.add_scalar('Eval/FID', fid, ep)
        writer.add_scalar('Eval/Diversity', diversity, ep)
        writer.add_scalar('Eval/top1', R_precision[0], ep)
        writer.add_scalar('Eval/top2', R_precision[1], ep)
        writer.add_scalar('Eval/top3', R_precision[2], ep)
        writer.add_scalar('Eval/matching_score', matching_score_pred, ep)


    draw = True
    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        if draw:print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            torch.save({"t2m_transformer":trans.state_dict(), "ep":ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if matching_score_pred > best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        if draw:print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        if draw:print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        if draw:print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        if draw:print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        if draw:print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [texts[k] for k in rand_idx]
        lengths = m_lengths[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)


    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching


@torch.no_grad()
def evaluation_mask_transformer_hml(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper,device, plot_func, time_steps = 10,
                           cond_scale=4, save_ckpt=False, save_anim=False):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        # m_length = m_length.cuda()
        # motions = motions.to(device).float().detach()
        m_length = m_length.to(device).long().detach()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # (b, seqlen)
        mids = trans.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)
        pred_motions = vq_model.forward_decoder(mids, m_length.clone())

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.to(device).float().detach()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    writer.add_scalar('Eval/FID', fid, ep)
    writer.add_scalar('Eval/Diversity', diversity, ep)
    writer.add_scalar('Eval/top1', R_precision[0], ep)
    writer.add_scalar('Eval/top2', R_precision[1], ep)
    writer.add_scalar('Eval/top3', R_precision[2], ep)
    writer.add_scalar('Eval/matching_score', matching_score_pred, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir,  'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)


    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching




@torch.no_grad()
def evaluation_momask(val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper, 
                      time_steps, cond_scale, temperature, topkr, gsample=True, 
                      force_mask=False, cal_mm=True, res_cond_scale=5):
    trans.eval()
    vq_model.eval()
    res_model.eval()

    device = res_model.device

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 1

    for i, batch in enumerate(tqdm(val_loader)):
        texts, motions, m_lengths = batch

        # motions = motions[..., :148]
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        et, _ = eval_wrapper.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_wrapper.encode_motion(motions[..., :148], m_lengths, sample_mean=True)
        bs, _ = motions.shape[0], motions.shape[1]

        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):

                mids = trans.generate(texts, m_lengths//4, time_steps, cond_scale, 
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                # motion_codes = motion_codes.permute(0, 2, 1)
                # mids.unsqueeze_(-1)
                pred_ids = res_model.generate(mids, texts, m_lengths//4, temperature=1, cond_scale=res_cond_scale)
                # pred_ids = mids.unsqueeze(-1)

                pred_motions = vq_model.forward_decoder(pred_ids)

                fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths, sample_mean=True)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(fid_em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(texts, m_lengths//4, time_steps, cond_scale, 
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

            pred_ids = res_model.generate(mids, texts, m_lengths//4, temperature=1, cond_scale=res_cond_scale)
            # pred_ids = mids.unsqueeze(-1)
            
            # pred_ids, _ = vq_model.encode(motions)
            pred_motions = vq_model.forward_decoder(pred_ids)

            # pred_motions[..., 1] = 0
            # motions[..., 90:100] = 0
            # pred_motions += torch.randn_like(pred_motions) 

            fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths, sample_mean=True)

        # pose = pose.cuda().float()
        motion_annotation_list.append(fid_em)
        motion_pred_list.append(fid_em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_momask_plus(val_loader, vq_model, trans, repeat_id, eval_wrapper, 
                      time_steps, cond_scale, temperature, topkr, gsample=True, cal_mm=True):
    trans.eval()
    vq_model.eval()

    device = trans.device

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 1
    else:
        num_mm_batch = 0

    for i, batch in enumerate(tqdm(val_loader)):
        texts, motions, m_lengths = batch

        # motions = motions[..., :148]
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        et, _ = eval_wrapper.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_wrapper.encode_motion(motions[..., :148], m_lengths, sample_mean=True)
        bs, _ = motions.shape[0], motions.shape[1]

        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):

                mids = trans.generate(texts, m_lengths//4, time_steps, cond_scale, 
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample)

                pred_motions = vq_model.forward_decoder(mids, m_lengths.clone())

                fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths, sample_mean=True)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(fid_em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(texts, m_lengths//4, time_steps, cond_scale, 
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample)

            pred_motions = vq_model.forward_decoder(mids, m_lengths.clone())

            fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths, sample_mean=True)

        # fid_em_pred, em_pred = fid_em, em
        # pose = pose.cuda().float()
        motion_annotation_list.append(fid_em)
        motion_pred_list.append(fid_em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_momask_plus_hml(val_loader, vq_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, cal_mm=True):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if  cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)
                pred_motions = vq_model.forward_decoder(mids, m_length.clone())

                # pred_motions = vq_model.decoder(codes)
                # pred_motions = vq_model.forward_decoder(mids)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)
            pred_motions = vq_model.forward_decoder(mids, m_length.clone())

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motions.clone(),
                                                              m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality