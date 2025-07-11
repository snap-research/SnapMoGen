import torch
from model.evaluator.modules import Encoder, EncoderV2, MLP
# from config.load_config import load_config
from model.encode_text import T5TextEncoder

from os.path import join as pjoin


def length_to_mask(length, max_len, device: torch.device = None):
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


class EvaluatorWrapper:
    def __init__(
        self,
        cfg,
        device,
        model_name='net_best_top1'
    ):
        # yaml_path = pjoin('./checkpoint_dir/snapmotion/evaluator', 
        #                   evaluator_name, 
        #                   'evaluator.yaml')
        # cfg = load_config(cfg_path)
        self.latent_enc = Encoder(
            nfeats = cfg.data.dim_pose,
            vae = cfg.vae,
            latent_dim = cfg.latent_encoder.latent_dim,
            ff_size = cfg.latent_encoder.ff_size,
            num_layers = cfg.latent_encoder.num_layers,
            num_heads = cfg.latent_encoder.num_heads,
            dropout = cfg.latent_encoder.dropout,
            activation = cfg.latent_encoder.activation,
        ) 


        if 't5' in cfg.text_embedder.version:
            self.text_enc = Encoder(
                nfeats = cfg.text_embedder.dim_embed,
                vae = cfg.vae,
                latent_dim = cfg.text_encoder.latent_dim,
                ff_size = cfg.text_encoder.ff_size,
                num_layers = cfg.text_encoder.num_layers,
                num_heads = cfg.text_encoder.num_heads,
                dropout = cfg.text_encoder.dropout,
                activation = cfg.text_encoder.activation,
            ) 

            self.text_emb = T5TextEncoder(
                device, 
            #  use_text_preprocessing,
                local_files_only=False, 
                from_pretrained=cfg.text_embedder.version, 
                model_max_length=cfg.data.max_text_length
                )
        elif 'ViT' in cfg.text_embedder.version:
            self.text_enc = MLP(
                nfeats = cfg.text_embedder.dim_embed, 
                vae = cfg.vae,
                latent_dim = cfg.text_encoder.latent_dim,
                ff_size = cfg.text_encoder.ff_size,
                dropout = cfg.text_encoder.dropout,
            )

            self.text_emb = CLIPTextEncoder(
                device=device,
                clip_version=cfg.text_embedder.version, 
            )
        
        model_path = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'evaluator', cfg.exp.name, 'model', '%s.tar'%model_name)
        model_weights = torch.load(model_path, map_location=device, weights_only=True)
        # print(model_weights.keys())

        self.text_enc.load_state_dict(model_weights['text_enc'])
        self.latent_enc.load_state_dict(model_weights['latent_enc'])
        self.text_enc.to(device)
        self.latent_enc.to(device)
        self.text_enc.eval()
        self.latent_enc.eval()
        self.ep = model_weights['ep']

        print("Load evaluation model from epoch %d!"%self.ep)

        # self.apply(init_weights)

    def eval(self):
        self.text_enc.eval()
        self.latent_enc.eval()

    def encode_text(self, text_input, sample_mean=True):
        text_embeddings, mask = self.text_emb.get_text_embeddings(text_input)
        # print(text_embeddings.shape, mask.shape)
        _, return_vecs, dist = self.text_enc.encode(text_embeddings, mask, sample_mean)
        return return_vecs, dist
    
    def encode_motion(self, motion_input, lengths, sample_mean=False):
        mask = length_to_mask(lengths, motion_input.shape[1], motion_input.device)
        fid_emb, return_vecs, dist = self.latent_enc.encode(motion_input, mask, sample_mean)
        return fid_emb, return_vecs, dist
    

class EvaluatorWrapperV2:
    def __init__(
        self,
        cfg,
        device,
        model_name='net_best_top1'
    ):
        # yaml_path = pjoin('./checkpoint_dir/snapmotion/evaluator', 
        #                   evaluator_name, 
        #                   'evaluator.yaml')
        # cfg = load_config(cfg_path)
        self.latent_enc = EncoderV2(
            nfeats = cfg.data.dim_pose,
            latent_dim = cfg.latent_encoder.latent_dim,
            output_dim = cfg.text_embedder.dim_embed,
            ff_size = cfg.latent_encoder.ff_size,
            num_layers = cfg.latent_encoder.num_layers,
            num_heads = cfg.latent_encoder.num_heads,
            dropout = cfg.latent_encoder.dropout,
            activation = cfg.latent_encoder.activation,
        ) 


        self.text_emb = CLIPTextEncoder(
            device=device,
            clip_version=cfg.text_embedder.version, 
        )
        
        model_path = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'evaluator', cfg.exp.name, 'model', '%s.tar'%model_name)
        model_weights = torch.load(model_path, map_location=device)

        self.latent_enc.load_state_dict(model_weights['latent_enc'])
        self.latent_enc.to(device)
        self.latent_enc.eval()
        self.ep = model_weights['ep']

        print("Load evaluation model from epoch %d!"%self.ep)

        # self.apply(init_weights)

    def eval(self):
        self.latent_enc.eval()

    def encode_text(self, text_input, sample_mean=False):
        text_embeddings, _ = self.text_emb.get_text_embeddings(text_input)
        # print(text_embeddings.shape, mask.shape)
        return text_embeddings, None
    
    def encode_motion(self, motion_input, lengths, sample_mean=False):
        mask = length_to_mask(lengths, motion_input.shape[1], motion_input.device)
        fid_emb, cst_vecs, rec_vecs = self.latent_enc.encode(motion_input, mask)
        return fid_emb, cst_vecs, rec_vecs