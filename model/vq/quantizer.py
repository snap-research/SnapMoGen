import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import random

# from vector_quantize_pytorch import ResidualVQ

#Borrow from vector_quantize_pytorch

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def length_to_mask(length, max_len=None, device: torch.device = None) -> torch.Tensor:
    if device is None:
        device = length.device

    if isinstance(length, list):
        length = torch.tensor(length)

    if max_len is None:
        max_len = max(length)
    
    length = length.to(device)
    # max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask

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


def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    dim = -1,
    training = True
):

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)

    return ind


class HRQuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu, scales):
        super(HRQuantizeEMAReset, self).__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        # print(nb_code, code_dim)
        self.mu = mu  ##TO_DO
        self.scales = scales
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, requires_grad=False))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def quantize(self, x, sample_codebook_temp=0.):
        # N X C -> C X N
        k_w = self.codebook.t()
        # x: NT X C
        # NT X N
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - \
                   2 * torch.matmul(x, k_w) + \
                   torch.sum(k_w ** 2, dim=0, keepdim=True)  # (N * L, b)

        # code_idx = torch.argmin(distance, dim=-1)

        code_idx = gumbel_sample(-distance, dim = -1, temperature = sample_codebook_temp, stochastic=True, training = self.training)

        return code_idx

    def dequantize(self, code_idx):
        mask = code_idx == -1.
        code_idx = code_idx.masked_fill(mask, 0.)

        x = F.embedding(code_idx, self.codebook)

        # x = x.masked_fill(mask, 0.)
        x[mask] = 0.
        return x
    
    def get_codebook_entry(self, indices):
        return self.dequantize(indices).permute(0, 2, 1)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        # print(x.shape, self.codebook.shape)
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device) # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)
        
        # print(code_onehot.shape, x.shape, self.codebook.shape)
        code_sum = torch.matmul(code_onehot, x) # nb_code, c
        code_count = code_onehot.sum(dim=-1) # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        if len(code_idx) > self.nb_code * 5:
            self.codebook = usage * code_update + (1-usage) * code_rand
        else:
            self.codebook = code_update

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity
    
    def quantize_all(self, x, m_lens=None, return_latent=False):
        N, width, T = x.shape

        residual = x.clone()
        f_hat = torch.zeros_like(x)
        idx_list = []

        for i, scale in enumerate(self.scales):

            if scale != 1:
                rest_down = F.interpolate(residual, size=int(T//scale), mode='area')
            else:
                rest_down = residual

            rest_down = rearrange(rest_down, 'n c t -> (n t) c')
            
            code_idx = self.quantize(rest_down)
            x_d = self.dequantize(code_idx)

            idx_list.append(rearrange(code_idx, '(n t) -> n t', n=N))

            x_d = rearrange(x_d, '(n t) c -> n c t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            residual -= up_x_d
            f_hat += up_x_d

            # if i == 8:
            #     break

        if return_latent:
            return idx_list, f_hat
        return f_hat
    
    def get_codes_from_indices(self, indices_list):
        assert len(indices_list) == len(self.scales)
        T = indices_list[-1].shape[-1]
        code = 0.0
        for indices, scale in zip(indices_list, self.scales):
            N, _ = indices.shape
            indices = rearrange(indices, 'n t->(n t)')
            x_d = self.dequantize(indices)
            x_d = rearrange(x_d, '(n t) d -> n d t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            code += up_x_d
        return code.permute(0, 2, 1)


    def forward(self, x, temperature=0., m_lens=None, quantize_dropout_prob=0.):
        N, width, T = x.shape

        residual = x.clone()
        f_hat = torch.zeros_like(x)
        mean_vq_loss = 0.
        # perplexity = 0.0

        if m_lens is not None:
            full_scale_mask = length_to_mask(m_lens, x.shape[-1])

        all_rest_down = []
        all_code_indices = []
        all_mask = []
        # should_quantize_dropout = self.training and random.random() < quantize_dropout_prob
        if self.training and random.random() < quantize_dropout_prob:
            start_drop_quantize_index = random.randint(1, len(self.scales)-1)
        else:
            start_drop_quantize_index = len(self.scales)
        for i, scale in enumerate(self.scales):
            if i >= start_drop_quantize_index:
                break
            # if should_quantize_dropout and 
            residual = residual * full_scale_mask.unsqueeze(1)
            if scale != 1:
                rest_down = F.interpolate(residual, size=int(T//scale), mode='area')
            else:
                rest_down = residual

            if m_lens is not None:
                mask = length_to_mask((m_lens//scale).long(), rest_down.shape[-1]) # (n t)
                all_mask.append(rearrange(mask, 'n t -> (n t)'))
                
            rest_down = rearrange(rest_down, 'n c t -> (n t) c')
            if self.training and not self.init:
                self.init_codebook(rest_down[all_mask[-1]])
            
            code_idx = self.quantize(rest_down, temperature)
            x_d = self.dequantize(code_idx)
            x_d[~all_mask[-1]] = 0

            all_rest_down.append(rest_down)
            all_code_indices.append(code_idx)
            
            x_d = rearrange(x_d, '(n t) c -> n c t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            residual -= up_x_d
            f_hat = f_hat + up_x_d

            if m_lens is not None:
                mean_vq_loss += mean_flat((x-f_hat.detach()).pow(2), full_scale_mask.unsqueeze(1))
                # F.mse_loss(x, f_hat, redu)
            else:
                mean_vq_loss += F.mse_loss(x, f_hat.detach())


        all_code_indices = torch.cat(all_code_indices, dim=0)
        all_rest_down = torch.cat(all_rest_down, dim=0)
        if m_lens is not None:
            all_mask = torch.cat(all_mask, dim=0)
            all_code_indices = all_code_indices[all_mask]
            all_rest_down = all_rest_down[all_mask]

        if self.training:
            perplexity = self.update_codebook(all_rest_down, all_code_indices)
        else:
            perplexity = self.compute_perplexity(all_code_indices)

        mean_vq_loss /= len(self.scales)
        # mean_perp /= len(self.scales)

        # f_hat = (f_hat.detach() - x) + x
        f_hat = x + (f_hat - x).detach()

        return f_hat, mean_vq_loss, perplexity
    
    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idx_to_var_input(self, indices_list):
        assert len(indices_list) == len(self.scales)
        T = indices_list[-1].shape[-1]
        code = 0.0
        next_scale_input = []
        for i in range(len(indices_list) - 1):
            indices = indices_list[i]
            N, _ = indices.shape
            indices = rearrange(indices, 'n t->(n t)')
            x_d = self.dequantize(indices)
            x_d = rearrange(x_d, '(n t) d -> n d t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            code += up_x_d
            next_scale = F.interpolate(code, size=int(T//self.scales[i+1]), mode='linear')
            next_scale_input.append(next_scale)
        return torch.cat(next_scale_input, dim=-1).permute(0, 2, 1)
    
    # ===================== get_next_var_input: only used in VAR inference, for getting next step's input =====================
    def get_next_var_input(self, level, indices, code, T):
        N, _ = indices.shape
        indices = rearrange(indices, 'n t -> (n t)')
        x_d = self.dequantize(indices)
        x_d = rearrange(x_d, '(n t) d -> n d t', n=N)
        
        if level != len(self.scales) - 1:
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            code += up_x_d
            next_scale = F.interpolate(code, size=int(T//self.scales[level+1]), mode='linear')
        else:
            code += x_d
            next_scale = code
        return code, next_scale
    


class HRQuantizeEMAResetV2(nn.Module):
    def __init__(self, nb_code, code_dim, mu, scales, share_quant_resi=4, quant_resi=0.5):
        super(HRQuantizeEMAResetV2, self).__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        # print(nb_code, code_dim)
        self.mu = mu  ##TO_DO
        self.scales = scales
        self.reset_codebook()
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:  # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared(
                [(Phi(code_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in
                 range(len(self.scales))])
        elif share_quant_resi == 1:  # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(code_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:   # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList(
                [(Phi(code_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in
                 range(share_quant_resi)]))


    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, requires_grad=False))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def quantize(self, x, sample_codebook_temp=0.):
        # N X C -> C X N
        k_w = self.codebook.t()
        # x: NT X C
        # NT X N
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - \
                   2 * torch.matmul(x, k_w) + \
                   torch.sum(k_w ** 2, dim=0, keepdim=True)  # (N * L, b)

        # code_idx = torch.argmin(distance, dim=-1)

        code_idx = gumbel_sample(-distance, dim = -1, temperature = sample_codebook_temp, stochastic=True, training = self.training)

        return code_idx

    def dequantize(self, code_idx):
        mask = code_idx == -1.
        code_idx = code_idx.masked_fill(mask, 0.)

        x = F.embedding(code_idx, self.codebook)

        # x = x.masked_fill(mask, 0.)
        x[mask] = 0.
        return x
    
    def get_codebook_entry(self, indices):
        return self.dequantize(indices).permute(0, 2, 1)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        # print(x.shape, self.codebook.shape)
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device) # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)
        
        # print(code_onehot.shape, x.shape, self.codebook.shape)
        code_sum = torch.matmul(code_onehot, x) # nb_code, c
        code_count = code_onehot.sum(dim=-1) # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        if len(code_idx) > self.nb_code * 5:
            self.codebook = usage * code_update + (1-usage) * code_rand
        else:
            self.codebook = code_update

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity
    
    def quantize_all(self, x, m_lens=None, return_latent=False):
        N, width, T = x.shape

        residual = x.clone()
        f_hat = torch.zeros_like(x)
        idx_list = []

        if m_lens is not None:
            full_scale_mask = length_to_mask(m_lens, x.shape[-1])
        else:
            full_scale_mask = torch.ones(x.shape[:-1], device=x.device).bool()

        for i, scale in enumerate(self.scales):

            residual = residual * full_scale_mask.unsqueeze(1)
                # all_mask.append(rearrange(mask, 'n t -> (n t)'))

            if scale != 1:
                rest_down = F.interpolate(residual, size=int(T//scale), mode='area')
            else:
                rest_down = residual

            if m_lens is not None:
                mask = length_to_mask((m_lens//scale).long(), rest_down.shape[-1]) # (n t)
                mask = rearrange(mask, 'n t -> (n t)')

            rest_down = rearrange(rest_down, 'n c t -> (n t) c')
            
            code_idx = self.quantize(rest_down)
            x_d = self.dequantize(code_idx)
            x_d[~mask] = 0

            code_idx[~mask] = -1
            idx_list.append(rearrange(code_idx, '(n t) -> n t', n=N))

            x_d = rearrange(x_d, '(n t) c -> n c t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            if len(self.scales) > 1:
                up_x_d = self.quant_resi[i / (len(self.scales) -1)](up_x_d)
            # up_x_d = self.quant_resi[i / (len(self.scales) -1)](up_x_d)

            residual -= up_x_d
            f_hat += up_x_d

        if return_latent:
            return idx_list, f_hat
        return idx_list
    
    def get_codes_from_indices(self, indices_list):
        assert len(indices_list) == len(self.scales)
        T = indices_list[-1].shape[-1]
        code = 0.0
        
        for i, (indices, scale) in enumerate(zip(indices_list, self.scales)):
            N, _ = indices.shape
            indices = rearrange(indices, 'n t->(n t)')
            x_d = self.dequantize(indices)
            x_d = rearrange(x_d, '(n t) d -> n d t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            if len(self.scales) > 1:
                up_x_d = self.quant_resi[i / (len(self.scales) -1)](up_x_d)
            code += up_x_d
        return code.permute(0, 2, 1)


    def forward(self, x, temperature=0., m_lens=None, start_drop=1, quantize_dropout_prob=0.):
        N, width, T = x.shape

        residual = x.clone()
        f_hat = torch.zeros_like(x)
        mean_vq_loss = 0.
        # perplexity = 0.0

        if m_lens is not None:
            full_scale_mask = length_to_mask(m_lens, x.shape[-1])

        all_rest_down = []
        all_code_indices = []
        all_mask = []
        
        if self.training and quantize_dropout_prob != 0:
             n_quantizers = torch.randint(start_drop, len(self.scales) + 1, (N, ))
             n_dropout = int(N * quantize_dropout_prob)
             n_quantizers[n_dropout:] = len(self.scales) + 1
             n_quantizers = n_quantizers.to(x.device)
        else:
            n_quantizers = torch.full((N,), len(self.scales)+1, device=x.device)

        for i, scale in enumerate(self.scales):
            
            # if should_quantize_dropout and 
            residual = residual * full_scale_mask.unsqueeze(1)

            keep_mask = (torch.full((N,), fill_value=i, device=x.device) < n_quantizers) # 1:keep, 0:drop

            if scale != 1:
                rest_down = F.interpolate(residual, size=int(T//scale), mode='area')
            else:
                rest_down = residual

            if m_lens is not None:
                mask = length_to_mask((m_lens//scale).long(), rest_down.shape[-1]) # (n t)
                mask = mask & keep_mask[:, None]
                all_mask.append(rearrange(mask, 'n t -> (n t)'))
                
            rest_down = rearrange(rest_down, 'n c t -> (n t) c')
            if self.training and not self.init:
                self.init_codebook(rest_down[all_mask[-1]])
            
            code_idx = self.quantize(rest_down, temperature)
            x_d = self.dequantize(code_idx)
            x_d[~all_mask[-1]] = 0.

            all_rest_down.append(rest_down)
            all_code_indices.append(code_idx)
            
            x_d = rearrange(x_d, '(n t) c -> n c t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            if len(self.scales) > 1:
                up_x_d = self.quant_resi[i / (len(self.scales) -1)](up_x_d)
            # up_x_d = self.quant_resi[i / (len(self.scales) -1)](up_x_d)
            up_x_d[~keep_mask] = 0.
            residual -= up_x_d
            f_hat = f_hat + up_x_d 

            if m_lens is not None:
                loss_mask = full_scale_mask & keep_mask[:, None]
                mean_vq_loss += mean_flat((x-f_hat.detach()).pow(2), loss_mask.unsqueeze(1))
                # F.mse_loss(x, f_hat, redu)
            else:
                # mean_vq_loss += F.mse_loss(x, f_hat.detach())
                mean_vq_loss += mean_flat((x-f_hat.detach()).pow(2), keep_mask[:, None, None])


        all_code_indices = torch.cat(all_code_indices, dim=0)
        all_rest_down = torch.cat(all_rest_down, dim=0)
        if m_lens is not None:
            all_mask = torch.cat(all_mask, dim=0)
            all_code_indices = all_code_indices[all_mask]
            all_rest_down = all_rest_down[all_mask]

        if self.training:
            perplexity = self.update_codebook(all_rest_down, all_code_indices)
        else:
            perplexity = self.compute_perplexity(all_code_indices)

        mean_vq_loss /= len(self.scales)
        # mean_perp /= len(self.scales)

        # f_hat = (f_hat.detach() - x) + x
        f_hat = x + (f_hat - x).detach()

        return f_hat, mean_vq_loss, perplexity
    
    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idx_to_var_input(self, indices_list):
        assert len(indices_list) == len(self.scales)
        T = indices_list[-1].shape[-1]
        code = 0.0
        next_scale_input = []
        for i in range(len(indices_list) - 1):
            indices = indices_list[i]
            N, _ = indices.shape
            indices = rearrange(indices, 'n t->(n t)')
            x_d = self.dequantize(indices)
            x_d = rearrange(x_d, '(n t) d -> n d t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            up_x_d = self.quant_resi[i / (len(self.scales) -1)](up_x_d)
            code += up_x_d
            next_scale = F.interpolate(code, size=int(T//self.scales[i+1]), mode='linear')
            next_scale_input.append(next_scale)
        return torch.cat(next_scale_input, dim=-1).permute(0, 2, 1)
    
    # ===================== get_next_var_input: only used in VAR inference, for getting next step's input =====================
    def get_next_var_input(self, level, indices, code, T):
        N, _ = indices.shape
        indices = rearrange(indices, 'n t -> (n t)')
        x_d = self.dequantize(indices)
        x_d = rearrange(x_d, '(n t) d -> n d t', n=N)
        
        if level != len(self.scales) - 1:
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            up_x_d = self.quant_resi[level / (len(self.scales) -1)](up_x_d)
            code += up_x_d
            next_scale = F.interpolate(code, size=int(T//self.scales[level+1]), mode='linear')
        else:
            code += x_d
            next_scale = code
        return code, next_scale


    
class Phi(nn.Conv1d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'