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
        input = input.view(n, -1, 2, x, y)
        output = super().forward(input)
        output = output.view(n, -1, x, y)
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
        input = input.view(n, -1, 2, x, y, z)
        output = super().forward(input)
        output = output.view(n, -1, x, y, z)
        return output


class LRInstanceNorm2d(_InstanceNorm):
    """Left-right reflection equivariant instance normalization 2D

    """
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        n, _, x, y = input.shape
        input = input.view(n, -1, 2, x, y)
        output = super().forward(input)
        output = output.view(n, -1, x, y)
        return output


class LRInstanceNorm3d(_InstanceNorm):
    """Left-right reflection equivariant instance normalization 3D

    """
    def _check_input_dim(self, input):
        if input.dim() != 6:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        n, _, x, y, z = input.shape
        input = input.view(n, -1, 2, x, y, z)
        output = super().forward(input)
        output = output.view(n, -1, x, y, z)
        return output
