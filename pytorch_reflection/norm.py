# -*- coding: utf-8 -*-

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


class LRBatchNorm2d(_BatchNorm):
    """Left-right reflection equivariant batch normalization 2D

    """
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        n, _, x, y = input.shape
        input = input.view(n, 2, -1, x, y).permute(0, 2, 1, 3, 4)
        output = super().forward(input)
        output = output.permute(0, 2, 1, 3, 4).contiguous().view(n, -1, x, y)
        return output


class LRBatchNorm3d(_BatchNorm):
    """Left-right reflection equivariant batch normalization 3D

    """
    def _check_input_dim(self, input):
        if input.dim() != 6:
            raise ValueError('expected 6D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        n, _, x, y, z = input.shape
        input = input.view(n, 2, -1, x, y, z).permute(0, 2, 1, 3, 4, 5)
        output = super().forward(input)
        output = output.permute(0, 2, 1, 3, 4, 5).contiguous()
        output = output.view(n, -1, x, y, z)
        return output


class _LRInstanceNorm(_InstanceNorm):
    """Left-right reflection equivariant instance normalization

    """
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        orig_weight = self.weight
        orig_bias = self.bias
        weight = torch.cat((self.weight, self.weight), 0)
        self.weight = torch.nn.Parameter(weight)
        bias = torch.cat((self.bias, self.bias), 0)
        self.bias = torch.nn.Parameter(bias)
        output = super().forward(input)
        self.weight = orig_weight
        self.bias = orig_bias
        return output


class LRInstanceNorm2d(_LRInstanceNorm):
    """Left-right reflection equivariant instance normalization 2D

    """
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class LRInstanceNorm3d(_LRInstanceNorm):
    """Left-right reflection equivariant instance normalization 3D

    """
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
