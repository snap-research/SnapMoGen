from torch.nn.utils import clip_grad_norm_
import torch
import numpy as np
from collections import OrderedDict

class BaseTrainer:
    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    @staticmethod
    def to(net_opt_list, device):
        for net_opt in net_opt_list:
            net_opt.to(device)

    @staticmethod
    def net_train(network_list):
        for network in network_list:
            network.train()

    @staticmethod
    def net_eval(network_list):
        for network in network_list:
            network.eval()

    @staticmethod
    def reparametrize(mu, logvar):
        s_var = logvar.mul(0.5).exp_()
        eps = s_var.data.new(s_var.size()).normal_()
        return eps.mul(s_var).add_(mu)

    @staticmethod
    def ones_like(tensor, val=1.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(tensor.device).requires_grad_(False)

    @staticmethod
    def zeros_like(tensor, val=0.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(tensor.device).requires_grad_(False)

    @staticmethod
    def kl_criterion(mu1, logvar1, mu2, logvar2):

        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (
                2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / np.prod(mu1.shape)

    @staticmethod
    def kl_criterion_unit(mu, logvar):
        kld = ((torch.exp(logvar) + mu ** 2) - logvar - 1) / 2
        return kld.sum() / np.prod(mu.shape)

    @staticmethod
    def swap(x):
        "Swaps the ordering of the minibatch"
        shape = x.shape
        assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
        new_shape = [shape[0]//2, 2] + list(shape[1:])
        x = x.view(*new_shape)
        x = torch.flip(x, [1])
        return x.view(*shape)
    
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr, opt_list):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for optimizer in opt_list:
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        return current_lr

    @staticmethod
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
        
    # return mask where padding is FALSE
    @staticmethod
    def lengths_to_mask(lengths, max_len=None):
        if max_len is None:
            max_len = max(lengths)
        mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        return mask #(b, len)
    
    @staticmethod
    @torch.no_grad()
    def update_ema(ema_model, model, decay=0.9999):
        """
        Update EMA model parameters towards the current model.
        """
        # for param in ema_model.parameters():
        #     param.requires_grad_(False)

        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())

        for name, param in model_params.items():
            if param.requires_grad:
                ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
            else:
                ema_params[name].copy_(param.data)

        for (ema_name, ema_buffer), (model_name, model_buffer) in zip(
        ema_model.named_buffers(), model.named_buffers()
    ):
            assert ema_name == model_name, f"Buffer name mismatch: {ema_name} vs {model_name}"
            ema_buffer.copy_(model_buffer)  # 直接复制

    @staticmethod
    def requires_grad(model, flag=True):
        """
        Set requires_grad flag for all parameters in a model.
        """
        for p in model.parameters():
            p.requires_grad = flag