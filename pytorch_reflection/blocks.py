# -*- coding: utf-8 -*-

from .conv import LRConvI2F2d, LRConvI2F3d, LRConvF2F2d, LRConvF2F3d
from .conv import LRConvF2I2d, LRConvF2I3d
from .norm import LRBatchNorm2d, LRBatchNorm3d
from .norm import LRInstanceNorm2d, LRInstanceNorm3d

import torch
from pytorch_engine import Config as EConfig
from pytorch_engine.layers import create_activ, create_dropout
from pytorch_engine.layers import create_interpolate, create_norm


def create_LRI2F(in_channels, out_channels, **kwargs):
    config = EConfig()
    if config.dim == 2:
        return LRConvI2F2d(in_channels, out_channels, 3, padding=1, **kwargs)
    elif config.dim == 3:
        return LRConvI2F3d(in_channels, out_channels, 3, padding=1, **kwargs)


def create_LRF2F(in_channels, out_channels, **kwargs):
    config = EConfig()
    if config.dim == 2:
        return LRConvF2F2d(in_channels, out_channels, 3, padding=1, **kwargs)
    elif config.dim == 3:
        return LRConvF2F3d(in_channels, out_channels, 3, padding=1, **kwargs)


def create_LRF2F_proj(in_channels, out_channels, **kwargs):
    config = EConfig()
    if config.dim == 2:
        return LRConvF2F2d(in_channels, out_channels, 1, padding=0, **kwargs)
    elif config.dim == 3:
        return LRConvF2F3d(in_channels, out_channels, 1, padding=0, **kwargs)


def create_LRF2I(in_channels, out_channels, **kwargs):
    config = EConfig()
    if config.dim == 2:
        return LRConvF2I2d(in_channels, out_channels, 1, padding=0, **kwargs)
    elif config.dim == 3:
        return LRConvF2I3d(in_channels, out_channels, 1, padding=0, **kwargs)


def create_LR_norm(num_features):
    config = EConfig()
    paras = config.norm.copy()
    paras.pop('name')
    if config.dim == 2:
        if config.norm['name'] == 'instance':
            return LRInstanceNorm2d(num_features, **paras)
        elif config.norm['name'] == 'batch':
            return LRBatchNorm2d(num_features, **paras)
    elif config.dim == 3:
        if config.norm['name'] == 'instance':
            return LRInstanceNorm3d(num_features, **paras)
        elif config.norm['name'] == 'batch':
            return LRBatchNorm3d(num_features, **paras)


class LRConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = self._create_conv()
        self.norm = create_LR_norm(out_channels)
        self.activ = create_activ()

    def _create_conv(self):
        raise NotImplementedError

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        output = self.activ(output)
        return output


class LRConvBlockI2F(LRConvBlock):
    def _create_conv(self):
        return create_LRI2F(self.in_channels, self.out_channels, bias=False)


class LRConvBlockF2F(LRConvBlock):
    def _create_conv(self):
        return create_LRF2F(self.in_channels, self.out_channels, bias=False)


class LRInputBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.conv1 = LRConvBlockI2F(in_channels, inter_channels)
        self.dp1 = create_dropout()
        self.conv2 = LRConvBlockF2F(inter_channels, out_channels)
        self.dp2 = create_dropout()

    def forward(self, input):
        output = self.conv1(input)
        output = self.dp1(output)
        output = self.conv2(output)
        output = self.dp2(output)
        return output


class LRContractingBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.conv1 = LRConvBlockF2F(in_channels, inter_channels)
        self.dp1 = create_dropout()
        self.conv2 = LRConvBlockF2F(inter_channels, out_channels)
        self.dp2 = create_dropout()
        
    def forward(self, input):
        output = self.conv1(input)
        output = self.dp1(output)
        output = self.conv2(output)
        output = self.dp2(output)
        return output


class LRExpandingBlock(torch.nn.Module):
    def __init__(self, in_channels, shortcut_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shortcut_channels = shortcut_channels

        in_channels = in_channels + shortcut_channels
        self.conv1 = LRConvBlockF2F(in_channels, out_channels)
        self.dp1 = create_dropout()
        self.conv2 = LRConvBlockF2F(out_channels, out_channels)
        self.dp2 = create_dropout()

    def forward(self, input, shortcut):
        output = torch.cat((input, shortcut), dim=1) # concat channels
        output = self.conv1(output)
        output = self.dp1(output)
        output = self.conv2(output)
        output = self.dp2(output)
        return output


class LRTransUpBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = create_LRF2F_proj(in_channels, out_channels, bias=False)
        self.norm = create_LR_norm(out_channels)
        self.activ = create_activ()
        self.up = create_interpolate(scale_factor=2)

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        output = self.activ(output)
        output = self.up(output)
        return output
