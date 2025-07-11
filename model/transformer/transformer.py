import torch
import torch.nn as nn
import numpy as np
# from networks.layers import *
import torch.nn.functional as F
# import clip
from model.encode_text import T5TextEncoder
from einops import repeat
from functools import partial
from model.transformer.tools import *
from torch.distributions.categorical import Categorical

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x

class PositionalEncoding(nn.Module):
    #Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, c, seqlen]
        return output

class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output
    

class MoMaskPlus(nn.Module):
    def __init__(self, code_dim, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, text_dim=512, cond_drop_prob=0.1,
                 device=None, cfg=None, full_length=80, scales=[8, 4, 2, 1]):
        super(MoMaskPlus, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.dropout = dropout
        self.cfg = cfg
        self.device = device
        self.full_length = full_length
        self.scales = scales
        self.patch_sizes = [int(full_length // scale) for scale in self.scales]
        self.cond_drop_prob = cond_drop_prob

        init_std = math.sqrt(1 / self.latent_dim / 3)


        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        if self.cfg.model.fuse_mode == 'in_context':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=num_heads,
                                                            dim_feedforward=ff_size,
                                                            dropout=dropout,
                                                            activation='gelu')

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=num_layers)
            
        elif self.cfg.model.fuse_mode == 'cross_attention':
            seqTransEncoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=num_heads,
                                                              dim_feedforward=ff_size,
                                                              dropout=dropout,
                                                              activation='gelu')
            self.seqTransEncoder = nn.TransformerDecoder(seqTransEncoderLayer,
                                                         num_layers=num_layers)
            
        self.lvl_embed = nn.Embedding(len(self.patch_sizes), self.latent_dim)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        # input_patch_size = self.patch_sizes

        self.cond_emb = nn.Linear(self.text_dim, self.latent_dim)

        _num_tokens = cfg.vq.nb_code + 2  # two dummy tokens, one for masking, one for padding
        self.mask_id = cfg.vq.nb_code
        self.pad_id = cfg.vq.nb_code + 1

        d = torch.cat([torch.full((ps,), i) for i, ps in enumerate(self.patch_sizes)]) #[1, 2, 2, 3, 3, 3, 3, 4, ...,]
        self.register_buffer('lvl_1L', d.contiguous())

        self.output_process = OutputProcess_Bert(out_feats=cfg.vq.nb_code, latent_dim=latent_dim)

        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)

        self.apply(self.__init_weights)

        '''
        Preparing frozen weights
        '''

        self.text_emb = T5TextEncoder(
            device, 
        #  use_text_preprocessing,
            local_files_only=False, 
            from_pretrained=cfg.text_embedder.version, 
            model_max_length=cfg.data.max_text_length
        )

        self.noise_schedule = cosine_schedule


    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def encode_text(self, raw_text):
        text_embedding, mask = self.text_emb.get_text_embeddings(raw_text)
        return text_embedding, mask
    
    def sinusoidal_encoding(self, t):
        """
        Compute sinusoidal positional encoding for a batch of timesteps t.
        Args:
            t (Tensor): Shape (B, L), representing the timestep indices.
            d_model (int): Embedding dimension.

        Returns:
            Tensor of shape (B, L, D).
        """
        div_term = torch.exp(torch.arange(0, self.latent_dim, 2, dtype=torch.float32, device=t.device) * (-math.log(10000.0) / self.latent_dim))
        
        pe = torch.zeros(*t.shape, self.latent_dim, device=t.device)  # (B, L, D)
        pe[..., 0::2] = torch.sin(t.unsqueeze(-1) * div_term)  # Apply sin to even indices
        pe[..., 1::2] = torch.cos(t.unsqueeze(-1) * div_term)  # Apply cos to odd indices
        
        return pe
    
    def get_pe_from_mlens(self, mlens, max_len):
        B = len(mlens)
        t = torch.arange(max_len, device=mlens.device).unsqueeze(0).expand(B, max_len) # [0, 1, 2, 3,..., max_len]
        T = mlens.unsqueeze(1).expand(B, max_len) # [12, 12, 12, 12, 12, ..., 12]
        t_progress = ((T - t - 1) / (T - 1 + 1e-4)) * 80 # [11/11, 10/11, 9/11, ..., 0/11] * 80
        torch.clamp_min_(t_progress, 0.)
        return self.sinusoidal_encoding(t_progress)


    def mask_cond(self, cond, force_mask=False):
        bs, _, _ =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1, 1)
            return cond * (1. - mask)
        else:
            return cond

    def trans_forward(self, motion_ids, cond, toa_pe, cond_padding_mask, motion_padding_mask):
        '''
        :param motion_ids: (b, seqlen)
        :cond_padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :motion_padding_mask: (b, t_seqlen), all pad positions are TRUE else FALSE
        :param cond: (b, t_seqlen, embed_dim) for text
        :param force_mask: boolean
        :return:
            -logits: (b, num_token, seqlen)
        '''
        b, t_seqlen, _ = cond.shape
        # cond = self.mask_cond(cond, force_mask=force_mask)

        # print(motion_ids.shape)
        x = self.token_emb(motion_ids)
        # print(x.shape)
        # (b, seqlen, d) -> (seqlen, b, latent_dim)
        x = self.input_process(x)

        cond = self.cond_emb(cond).permute(1, 0, 2) #(t, b, latent_dim)

        x = self.position_enc(x)
        cond = self.position_enc(cond)

        if self.cfg.model.use_toa_pe:
            x = x + toa_pe.permute(1, 0, 2)

        if self.cfg.model.use_lvl_pe:
            x = x + self.lvl_embed(self.lvl_1L).unsqueeze(1) 

        if self.cfg.model.fuse_mode == 'in_context':
            xseq = torch.cat([cond, x], dim=0) #(seqlen+t_seqlen, b, latent_dim)
            padding_mask = torch.cat([cond_padding_mask, motion_padding_mask], dim=1).bool() #(b, seqlen+t_seqlen)
            output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[t_seqlen:] #(seqlen, b, e)
            logits = self.output_process(output) #(seqlen, b, e) -> (b, ntoken, seqlen)
        elif self.cfg.model.fuse_mode == 'cross_attention':
            output = self.seqTransEncoder(x, 
                                          cond, 
                                          tgt_key_padding_mask=motion_padding_mask, 
                                          memory_key_padding_mask=cond_padding_mask,
                                          )
            logits = self.output_process(output) #(seqlen, b, e) -> (b, ntoken, seqlen)
        return logits

    def forward(self, id_list, y, m_lens):
        '''
        :param ids: (b, n)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        '''

        # ids = []
        non_pad_mask = []
        ids = []
        time_to_arrival_pe = []
        assert self.full_length == id_list[-1].shape[1]
        for scale, ele in zip(self.scales, id_list):
            ds_mlens = (m_lens // scale).long() 
            ds_non_pad_mask = lengths_to_mask(ds_mlens, ele.shape[1])
            non_pad_mask.append(ds_non_pad_mask)
            ids.append(ele)
            time_to_arrival_pe.append(self.get_pe_from_mlens(ds_mlens, ele.shape[1]))

        
        ids = torch.cat(ids, dim=1)
        non_pad_mask = torch.cat(non_pad_mask, dim=1)

        assert ids.shape[:2] == non_pad_mask.shape[:2]

        bs, ntokens = ids.shape
        time_to_arrival_pe = torch.cat(time_to_arrival_pe, dim=1)
        device = ids.device

        # Positions that are PADDED are ALL FALSE
        # non_pad_mask = lengths_to_mask(m_lens, ntokens) #(b, n)
        ids = torch.where(non_pad_mask, ids, self.pad_id)

        with torch.no_grad():
            cond_embs, cond_att_mask = self.encode_text(y)
            cond_padding_mask = (cond_att_mask==0)

        '''
        Prepare mask
        '''
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        # Positions to be MASKED are ALL TRUE
        mask = batch_randperm < num_token_masked.unsqueeze(-1)

        # Positions to be MASKED must also be NON-PADDED
        mask &= non_pad_mask

        # Note this is our training target, not input
        labels = torch.where(mask, ids, self.mask_id)

        x_ids = ids.clone()

        # Further Apply Bert Masking Scheme
        # Step 1: 10% replace with an incorrect token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_ids, high=self.cfg.vq.nb_code)
        x_ids = torch.where(mask_rid, rand_id, x_ids)
        # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)

        # mask_mid = mask

        x_ids = torch.where(mask_mid, self.mask_id, x_ids)

        if self.training and self.cfg.training.pert_prob > 0.:
            mask_rid = get_mask_subset_prob(~mask & non_pad_mask, self.cfg.training.pert_prob)
            rand_tokens = torch.randint_like(x_ids, high=self.cfg.vq.nb_code)
            x_ids = torch.where(mask_rid, rand_tokens, x_ids)

        cond_embs = self.mask_cond(cond_embs)

        logits = self.trans_forward(x_ids, cond_embs, time_to_arrival_pe, cond_padding_mask, ~non_pad_mask)
        ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)

        return ce_loss, pred_id, acc

    def forward_with_cond_scale(self,
                                motion_ids,
                                cond_embs,
                                time_to_arrival_pe,
                                cond_padding_mask,
                                motion_padding_mask,
                                cond_scale=3):
        # bs = motion_ids.shape[0]
        # if cond_scale == 1:

        input_motion_ids = torch.cat([motion_ids, motion_ids], dim=0)
        input_cond_embs = torch.cat([self.mask_cond(cond_embs, force_mask=True),
                                     self.mask_cond(cond_embs, force_mask=False)], dim=0)
        input_cond_padding_mask = torch.cat([cond_padding_mask, cond_padding_mask], dim=0)
        input_motion_padding_mask = torch.cat([motion_padding_mask, motion_padding_mask], dim=0)
        input_toa_pe = torch.cat([time_to_arrival_pe, time_to_arrival_pe], dim=0)

        output_logits = self.trans_forward(input_motion_ids, 
                                           input_cond_embs, 
                                           input_toa_pe,
                                           input_cond_padding_mask, 
                                           input_motion_padding_mask)
        aux_logits, logits = output_logits.chunk(2, dim=0)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 topk_filter_thres=0.9,
                 gsample=False,
                #  scales=[8, 4, 2, 1],
                 ):
        # print(self.cfg.vq.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.cfg.vq.num_quantizers

        device = next(self.parameters()).device
        seq_len = max(m_lens)
        # batch_size = len(m_lens)
        non_padding_mask = []
        lengths_div = []
        new_mlens = torch.zeros_like(m_lens)
        time_to_arrival_pe = []
        for scale in self.scales:
            non_padding_mask.append(
                lengths_to_mask((m_lens//scale).long(), int(self.full_length//scale))
            )
            lengths_div.append(int(self.full_length//scale))
            new_mlens += m_lens // scale
            time_to_arrival_pe.append(self.get_pe_from_mlens((m_lens//scale).long(), int(self.full_length//scale)))

        non_padding_mask = torch.cat(non_padding_mask, dim=1)
        padding_mask = ~non_padding_mask
        time_to_arrival_pe = torch.cat(time_to_arrival_pe, dim=1)

        with torch.no_grad():
            cond_embs, cond_att_mask = self.encode_text(conds)
            cond_padding_mask = (cond_att_mask==0)


        # padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # print(padding_mask.shape, )

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.)
        starting_temperature = temperature

        for timestep in torch.linspace(0, 1, timesteps, device=device):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * new_mlens).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(ids, 
                                                  cond_embs, 
                                                  time_to_arrival_pe=time_to_arrival_pe,
                                                  cond_padding_mask=cond_padding_mask,
                                                  motion_padding_mask=padding_mask,
                                                  cond_scale=cond_scale)
            

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # print(logits.shape, self.cfg.num_tokens)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            temperature = starting_temperature
            if gsample:  # use gumbel_softmax sampling
                # print("1111")
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                # print("2222")
                probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)


            ids = torch.where(is_mask, pred_ids, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        return_list = []
        start = 0
        for length in lengths_div:
            return_list.append(ids[..., start:start+length])
            start += length
        # print("Final", ids.max(), ids.min())
        return return_list