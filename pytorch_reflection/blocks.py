# -*- coding: utf-8 -*-

from .conv import LRConvI2F2d, LRConvI2F3d, LRConvF2F2d, LRConvF2F3d
from .conv import LRConvF2I2d, LRConvF2I3d
from .norm import LRBatchNorm2d, LRBatchNorm3d
from .norm import LRInstanceNorm2d, LRInstanceNorm3d

import torch
from pytorch_engine import Config as EConfig
from pytorch_engine.layers import create_activ, create_dropout
from pytorch_engine.layers import create_interpolate, create_norm
from pytorch_engine.layers import create_three_conv, create_proj


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


class _ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = self._create_conv()
        self.norm = self._create_norm()
        self.activ = create_activ()

    def _create_conv(self):
        raise NotImplementedError

    def _create_norm(self):
        raise NotImplementedError

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        output = self.activ(output)
        return output


class _LRConvBlock(_ConvBlock):
    def _create_norm(self):
        return create_LR_norm(self.out_channels)


class ConvBlock(_ConvBlock):
    def _create_conv(self):
        return create_three_conv(self.in_channels, self.out_channels,bias=False)

    def _create_norm(self):
        return create_norm(self.out_channels)


class LRConvBlockI2F(_LRConvBlock):
    def _create_conv(self):
        return create_LRI2F(self.in_channels, self.out_channels, bias=False)


class LRConvBlockF2F(_LRConvBlock):
    def _create_conv(self):
        return create_LRF2F(self.in_channels, self.out_channels, bias=False)


class _InputBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, inter_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.conv1 = self._create_conv1()
        self.dp1 = create_dropout()
        self.conv2 = self._create_conv2()
        self.dp2 = create_dropout()

    def forward(self, input):
        # print('ib input', input.shape)
        output = self.conv1(input)
        # print('ib after conv1', output.shape)
        output = self.dp1(output)
        output = self.conv2(output)
        # print('ib after conv2', output.shape)
        output = self.dp2(output)
        return output

    def _create_conv1(self):
        raise NotImplementedError

    def _create_conv2(self):
        raise NotImplementedError


class InputBlock(_InputBlock):

    def _create_conv1(self):
        return ConvBlock(self.in_channels, self.inter_channels)

    def _create_conv2(self):
        return ConvBlock(self.inter_channels, self.out_channels)


class LRInputBlock(_InputBlock):

    def _create_conv1(self):
        return LRConvBlockI2F(self.in_channels, self.inter_channels)

    def _create_conv2(self):
        return LRConvBlockF2F(self.inter_channels, self.out_channels)


class _ContractingBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, inter_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.conv1 = self._create_conv1()
        self.dp1 = create_dropout()
        self.conv2 = self._create_conv2()
        self.dp2 = create_dropout()
        
    def forward(self, input):
        # print('cb input', input.shape)
        output = self.conv1(input)
        # print('cb after conv1', output.shape)
        output = self.dp1(output)
        output = self.conv2(output)
        # print('cb after conv2', output.shape)
        output = self.dp2(output)
        return output

    def _create_conv1(self):
        raise NotImplementedError

    def _create_conv2(self):
        raise NotImplementedError


class ContractingBlock(_ContractingBlock):
    def _create_conv1(self):
        return ConvBlock(self.in_channels, self.inter_channels)

    def _create_conv2(self):
        return ConvBlock(self.inter_channels, self.out_channels)


class LRContractingBlock(_ContractingBlock):
    def _create_conv1(self):
        return LRConvBlockF2F(self.in_channels, self.inter_channels)

    def _create_conv2(self):
        return LRConvBlockF2F(self.inter_channels, self.out_channels)


class _ExpandingBlock(torch.nn.Module):
    def __init__(self, in_channels, shortcut_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shortcut_channels = shortcut_channels

        in_channels = in_channels + shortcut_channels
        self.conv1 = self._create_conv1()
        self.dp1 = create_dropout()
        self.conv2 = self._create_conv2()
        self.dp2 = create_dropout()

    def forward(self, input, shortcut):
        # print('eb input', input.shape)
        # print('eb shortcut', shortcut.shape)
        output = torch.cat((input, shortcut), dim=1) # concat channels
        output = self.conv1(output)
        # print('eb after conv1', output.shape)
        output = self.dp1(output)
        output = self.conv2(output)
        # print('eb after conv2', output.shape)
        output = self.dp2(output)
        return output

    def _create_conv1(self):
        raise NotImplementedError

    def _create_conv2(self):
        raise NotImplementedError


class ExpandingBlock(_ExpandingBlock):

    def _create_conv1(self):
        in_channels = self.in_channels + self.shortcut_channels
        return ConvBlock(in_channels, self.out_channels)

    def _create_conv2(self):
        return ConvBlock(self.out_channels, self.out_channels)


class LRExpandingBlock(_ExpandingBlock):

    def _create_conv1(self):
        in_channels = self.in_channels + self.shortcut_channels
        return LRConvBlockF2F(in_channels, self.out_channels)

    def _create_conv2(self):
        return LRConvBlockF2F(self.out_channels, self.out_channels)


class _TransUpBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = self._create_conv()
        self.norm = self._create_norm()
        self.activ = create_activ()
        self.dp = create_dropout()
        self.up = create_interpolate(scale_factor=2)

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        output = self.activ(output)
        output = self.dp(output)
        output = self.up(output)
        return output

    def _create_conv(self):
        raise NotImplementedError

    def _create_norm(self):
        raise NotImplementedError


class TransUpBlock(_TransUpBlock):

    def _create_conv(self):
        return create_proj(self.in_channels, self.out_channels, bias=False)

    def _create_norm(self):
        return create_norm(self.out_channels)


class LRTransUpBlock(_TransUpBlock):

    def _create_conv(self):
        return create_LRF2F_proj(self.in_channels, self.out_channels,bias=False)

    def _create_norm(self):
        return create_LR_norm(self.out_channels)
