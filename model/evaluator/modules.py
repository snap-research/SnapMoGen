from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from einops import repeat
# from model.encode_text import T5TextEncoder

# import torch.nn.functional as F



def length_to_mask(length, max_len, device: torch.device = None) -> Tensor:
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

def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0, std=1)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def reparametrize(mu, logvar):
    s_var = logvar.mul(0.5).exp_()
    eps = s_var.data.new(s_var.size()).normal_()
    return eps.mul(s_var).add_(mu)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False) -> None:
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class Encoder(nn.Module):
    # Similar to ACTOR but "action agnostic" and more general
    def __init__(
        self,
        nfeats: int,
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.nfeats = nfeats
        self.projection = nn.Linear(nfeats, latent_dim)

        self.vae = vae
        self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout=dropout, batch_first=True
        )

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

        self.linear = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        

        self.apply(init_weights)

    def forward(self, x_dict: Dict) -> Tensor:
        x = x_dict["x"]
        mask = x_dict["mask"]

        # print(x.shape, mask.shape)

        x = self.projection(x)

        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)

        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device)
        aug_mask = torch.cat((token_mask, mask), 1).bool()
        # print(aug_mask)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        # print(xseq.shape, aug_mask.shape)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        return final[:, 0], self.linear(final[:, : self.nbtokens])
    

    def encode(self, input, mask, sample_mean=False):
        fid_emb, output = self.forward({"x":input, "mask":mask})
        return_vec = None
        if self.vae:
            dists = output.unbind(1)
            mu, logvar = dists
            # print(mu.min(), mu.max(), mu.mean())
            logvar = torch.clamp(logvar, -10.0, 10.0)
            if sample_mean:
                return_vec = mu
            else:
                return_vec = reparametrize(mu, logvar)
        else:
            (return_vec, ) = output.unbind(1)
            dists = None
        return fid_emb, return_vec, dists
    

class EncoderV2(nn.Module):
    # Similar to ACTOR but "action agnostic" and more general
    def __init__(
        self,
        nfeats: int,
        latent_dim: int = 256,
        output_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.nfeats = nfeats
        self.projection = nn.Linear(nfeats, latent_dim)

        self.tokens = nn.Parameter(torch.randn(1, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout=dropout, batch_first=True
        )

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

        self.rec_linear = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.cst_linear = nn.Linear(in_features=latent_dim, out_features=output_dim)

        self.apply(init_weights)


    def encode(self, input, mask) -> Tensor:
        x = input
        # mask = x_dict["mask"]

        # print(x.shape, mask.shape)

        x = self.projection(x)

        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)

        token_mask = torch.ones((bs, 1), dtype=bool, device=device)
        aug_mask = torch.cat((token_mask, mask), 1).bool()
        # print(aug_mask)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        # print(xseq.shape, aug_mask.shape)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        return final[:, 0], self.cst_linear(final[:, 0]), self.rec_linear(final[:, 0])
    

    # def encode(self, input, mask):

    #     return  self.forward({"x":input, "mask":mask})
    


class MLP(nn.Module):
    def __init__(
        self,
        nfeats: int,
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.vae = vae
        self.model = nn.Sequential(
            nn.Linear(nfeats, ff_size),
            nn.LayerNorm(ff_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(ff_size, ff_size),
            nn.LayerNorm(ff_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(ff_size, ff_size),
            nn.LayerNorm(ff_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(ff_size, latent_dim * 2)
        )

        self.apply(init_weights)

    def forward(self, x_dict: Dict) -> Tensor:

        return self.model(x_dict["x"])
    

    def encode(self, input, mask, sample_mean=False):
        output = self.forward({"x":input})
        return_vec = None
        if self.vae:
            dists = output.chunk(2, dim=-1)
            mu, logvar = dists
            # print(mu.min(), mu.max(), mu.mean())
            logvar = torch.clamp(logvar, -10.0, 10.0)
            if sample_mean:
                return_vec = mu
            else:
                return_vec = reparametrize(mu, logvar)
        else:
            return_vec = output
            dists = None
        return return_vec, dists




class Decoder(nn.Module):
    # Similar to ACTOR Decoder

    def __init__(
        self,
        nfeats: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        output_feats = nfeats
        self.nfeats = nfeats
        
        self.linear = nn.Linear(in_features=latent_dim, out_features=latent_dim)

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout, batch_first=True
        )

        seq_trans_decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            layer_norm_eps=1e-5
        )

        self.seqTransDecoder = nn.TransformerDecoder(
            seq_trans_decoder_layer, num_layers=num_layers
        )

        # for layer in self.seqTransDecoder.layers:
        #     layer.register_forward_hook(debug_hook)


        self.final_layer = nn.Linear(latent_dim, output_feats)
        self.apply(init_weights)

    def forward(self, z_dict: Dict) -> Tensor:
        z = z_dict["z"]
        mask = z_dict["mask"]

        latent_dim = z.shape[-1]
        bs, nframes = mask.shape


        z = z[:, None, :]  # sequence of 1 element for the memory
        z = self.linear(z)

        time_queries = torch.zeros(bs, nframes, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)


        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(
            tgt=time_queries, memory=z, tgt_key_padding_mask=~mask
        )

        output = self.final_layer(output)
        output[~mask] = 0

        return output
    


class Evaluator(nn.Module):
    def __init__(
        self,
        latent_enc: nn.Module,
        latent_dec: nn.Module,
        text_enc: nn.Module,
        text_emb: nn.Module,
        vae: bool,
    ):
        super().__init__()

        self.latent_enc = latent_enc
        self.latent_dec = latent_dec
        self.text_enc = text_enc
        self.text_emb = text_emb

        self.vae = vae

        # self.apply(init_weights)

    def encode_text(self, text_input, sample_mean=False):
        text_embeddings, mask = self.text_emb.get_text_embeddings(text_input)
        # print(text_embeddings.shape, mask.shape)
        _, return_vecs, dist = self.text_enc.encode(text_embeddings, mask, sample_mean)
        return return_vecs, dist
    
    def encode_motion(self, motion_input, lengths, sample_mean=False):
        mask = length_to_mask(lengths, motion_input.shape[1], motion_input.device)
        fid_emb, return_vecs, dist = self.latent_enc.encode(motion_input, mask, sample_mean)
        return fid_emb, return_vecs, dist
    
    def decode(self, latent_vectors, max_length, lengths):
        mask = length_to_mask(lengths, max_length, device=latent_vectors.device)
        z_dict = {"z": latent_vectors, "mask": mask}
        motions = self.latent_dec(z_dict)
        return motions
    

class EvaluatorV2(nn.Module):
    def __init__(
        self,
        latent_enc: nn.Module,
        latent_dec: nn.Module,
        text_emb: nn.Module,
    ):
        super().__init__()

        self.latent_enc = latent_enc
        self.latent_dec = latent_dec
        self.text_emb = text_emb

        # self.apply(init_weights)

    def encode_text(self, text_input, sample_mean=False):
        text_embeddings, _ = self.text_emb.get_text_embeddings(text_input)
        # print(text_embeddings.shape, mask.shape)
        return text_embeddings, None
    
    def encode_motion(self, motion_input, lengths, sample_mean=False):
        mask = length_to_mask(lengths, motion_input.shape[1], motion_input.device)
        fid_emb, cst_vecs, rec_vecs = self.latent_enc.encode(motion_input, mask)
        return fid_emb, cst_vecs, rec_vecs
    
    def decode(self, latent_vectors, max_length, lengths):
        mask = length_to_mask(lengths, max_length, device=latent_vectors.device)
        z_dict = {"z": latent_vectors, "mask": mask}
        motions = self.latent_dec(z_dict)
        return motions
    

# class EvaluatorWrapper(nn.Module):
#     def __init__(
#         self,
#         text_emb: nn.Module,
#         text_enc: nn.Module,

#         motion_enc: nn.Module,
#         latent_enc: nn.Module,
#         latent_dec=None,
#         motion_dec=None,
#         unit_length=1,
#     ):
#         super().__init__()
#         self.text_emb = text_emb
#         self.text_enc = text_enc
#         self.motion_enc = motion_enc
#         self.latent_enc = latent_enc
#         self.latent_dec = latent_dec
#         self.unit_length = unit_length
#         self.motion_dec = motion_dec

#         # self.apply(init_weights)


#     def encode_text(self, text_input, sample_mean=False):
#         text_embeddings, mask = self.text_emb.get_text_embeddings(text_input)
#         # print(text_embeddings.shape, mask.shape)
#         return_vecs, dist = self.text_enc.encode(text_embeddings, mask, sample_mean)
#         return return_vecs, dist
    
#     def encode_motion(self, motion_input, lengths, sample_mean=False):
        
#         # motion_input = motion_input.permute(0, 2, 1)
#         latent_input = self.motion_enc(motion_input)
#         lengths //= self.unit_length
#         mask = length_to_mask(lengths, latent_input.shape[1], latent_input.device)
#         return_vecs, dist = self.latent_enc.encode(latent_input, mask, sample_mean)
#         return return_vecs, dist
    
#     def decode(self, latent_vectors, max_length, lengths):
#         assert self.latent_dec is not None
#         assert self.motion_dec is not None
#         trans_max_length = max_length // self.unit_length 
#         trans_lengths = lengths // self.unit_length
#         trans_mask = length_to_mask(trans_lengths, trans_max_length, device=latent_vectors.device)
#         z_dict = {"z": latent_vectors, "mask": trans_mask}
#         motion_latents = self.latent_dec(z_dict)

#         mask = length_to_mask(lengths, max_length, device=latent_vectors.device)
#         motions = self.motion_dec(motion_latents)
#         # print(motions.shape)
#         motions[~mask] = 0

#         return motions