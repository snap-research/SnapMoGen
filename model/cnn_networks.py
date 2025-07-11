import torch
import torch.nn as nn
import torch.nn.init as init
from model.blocks import Resnet1D, SimpleConv1dLayer, Conv1dLayer


def length_to_mask(length, max_len=None, device: torch.device = None):
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

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        # 使用 Xavier 初始化
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # 使用 Xavier 初始化
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def reparametrize(mu, logvar):
    s_var = logvar.mul(0.5).exp_()
    eps = s_var.data.new(s_var.size()).normal_()
    return eps.mul(s_var).add_(mu)

class GlobalRegressor(nn.Module):
    def __init__(self, dim_in, dim_latent, dim_out):
        super().__init__()
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv1d(dim_in, dim_latent, 3, 1, 1),
                nn.LeakyReLU(0.2)
                )
        )
        
        layers.append(Resnet1D(dim_latent, n_depth=3, dilation_growth_rate=3, reverse_dilation=True))
        # layers.append(Resnet1D(dim_latent, n_depth=2, dilation_growth_rate=3, reverse_dilation=True))
        layers.append(nn.Conv1d(dim_latent, dim_out, 3, 1, 1))
        self.layers = nn.Sequential(*layers)
        self.apply(init_weights)



    def forward(self, input):
        input = input.permute(0, 2, 1)
        return self.layers(input).permute(0, 2, 1)
    


############################################
################# VQ Model #################
############################################
class EncoderAttn(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 use_attn=False,
                 norm=None):
        super().__init__()

        filter_t, pad_t = stride_t * 2, stride_t // 2
        self.embed = nn.Sequential(
            nn.Conv1d(input_emb_width, width, 3, 1, 1),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            self.res_blocks.append(block)
            self.attn_blocks.append(make_attn(width, use_attn=use_attn))
        self.outproj = nn.Conv1d(width, output_emb_width, 3, 1, 1)
        # blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        # self.model = nn.Sequential(*blocks)
        self.apply(init_weights)

    def forward(self, x, m_lens=None):
        x = self.embed(x)
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x)
            if m_lens is not None: m_lens = m_lens//2
            x = attn_block(x, m_lens)
        return self.outproj(x)


class DecoderAttn(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 use_attn = False,
                 norm=None):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv1d(output_emb_width, width, 3, 1, 1),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            self.res_blocks.append(block)
            self.attn_blocks.append(make_attn(width, use_attn))

        self.outproj = nn.Sequential(
            nn.Conv1d(width, width, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(width, input_emb_width, 3, 1, 1)
        )
        self.apply(init_weights)

    def forward(self, x, m_lens=None, keep_shape=False):
        x = self.embed(x)

        # m_lens //= 2**len(self.res_blocks)
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x)
            if m_lens is not None: m_lens *= 2
            x = attn_block(x, m_lens)

        x = self.outproj(x)

        if keep_shape:
            return x
        else:
            return x.permute(0, 2, 1)


def make_attn(in_channels, use_attn=True):
    return AttnBlock(in_channels) if use_attn else MultiInputIdentity()


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn_block = nn.MultiheadAttention(in_channels, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, m_lens):
        x = x.permute(0, 2, 1)
        key_mask = length_to_mask(m_lens, x.shape[1])

        attn_out, _ = self.attn_block(
            self.norm(x), self.norm(x), self.norm(x), key_padding_mask = ~key_mask
        )

        x = x + attn_out
        return x.permute(0, 2, 1)
    

class MultiInputIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, m_lens=None):
        return x