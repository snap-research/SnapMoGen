import torch.nn as nn
import torch
from collections import OrderedDict

class nonlinearity(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    
class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=0.2):
        super(ResConv1DBlock, self).__init__()

        padding = dilation
        self.norm = norm

        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
        elif activation == "leakyrelu":
            self.activation1 = nn.LeakyReLU(0.1)
            self.activation2 = nn.LeakyReLU(0.1)

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0, )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = self.dropout(x)
        x = x + x_orig
        return x
    

class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=3, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()

        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
    

class Conv1dLayer(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, bias=True, drop_prob=0, activate=True, norm="none"):
        super().__init__()
        layers = []

        padding = kernel_size // 2

        if padding > 0:
            layers.append(("RefPad", nn.ReflectionPad1d(padding)))
            padding = 0
        
        if downsample:
            stride = 2
        else:
            stride =1

        layers.append(("Conv", nn.Conv1d(in_channels=in_channel,
                                         out_channels=out_channel,
                                         kernel_size=kernel_size,
                                         padding=padding,
                                         stride=stride,
                                         bias=bias)))
        if norm == "bn":
            layers.append(("BatchNorm", nn.BatchNorm1d(out_channel)))
        elif norm == "in":
            layers.append(("InstanceNorm", nn.InstanceNorm1d(out_channel)))
        elif norm == "none":
            pass
        else:
            raise "Unsupported normalization:{}".format(norm)
        
        if drop_prob>0:
            layers.append(("Drop", nn.Dropout(drop_prob, inplace=True)))
        
        if activate:
            layers.append(("Act", nn.LeakyReLU(0.2, inplace=True)))
        super().__init__(OrderedDict(layers))
        self.apply(init_weight)

    def forward(self, input):
        out = super().forward(input)
        return out
    

class SimpleConv1dLayer(nn.Module):
    def __init__(self, in_channel, out_channel, upsample=True):
        super().__init__()

        if upsample:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1)
            )
        else:
            self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input):
        out = self.conv1(input)
        out = self.lrelu1(out)
        return out