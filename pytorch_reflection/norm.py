# -*- coding: utf-8 -*-

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
        num_samples, num_channels, x, y = input.shape
        input = input.view(num_samples, 2, -1, x, y).permute(0, 2, 1, 3, 4)
        return super().forward(input)


class LRBatchNorm3d(_BatchNorm):
    """Left-right reflection equivariant batch normalization 3D

    """
    def _check_input_dim(self, input):
        if input.dim() != 6:
            raise ValueError('expected 6D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        num_samples, num_channels, x, y, z = input.shape
        input = input.view(num_samples, 2, -1, x, y, z)
        input = input.permute(0, 2, 1, 3, 4, 5)
        return super().forward(input)


class LRInstanceNorm2d(_InstanceNorm):
    """Left-right reflection equivariant instance normalization 2D

    """
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        num_samples, num_channels, x, y = input.shape
        input = input.view(num_samples, 2, -1, x, y).permute(0, 2, 1, 3, 4)
        return super().forward(input)


class LRInstanceNorm3d(_InstanceNorm):
    """Left-right reflection equivariant instance normalization 3D

    """
    def _check_input_dim(self, input):
        if input.dim() != 6:
            raise ValueError('expected 6D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        num_samples, num_channels, x, y, z = input.shape
        input = input.view(num_samples, 2, -1, x, y, z)
        input = input.permute(0, 2, 1, 3, 4, 5)
        return super().forward(input)
