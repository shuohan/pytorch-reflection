# -*- coding: utf-8 -*-

import torch
from torch.nn.modules.batchnorm import BatchNorm2d, BatchNorm3d
from torch.nn.modules.instancenorm import InstanceNorm2d, InstanceNorm3d

from .conv import interleave_bias


class LRBatchNorm2d(BatchNorm2d):
    """Left-right reflection equivariant batch normalization 2D

    """
    def forward(self, input):
        n, _, x, y = input.shape
        input = input.view(n, -1, 2 * x, y)
        output = super().forward(input)
        output = output.view(n, -1, x, y)
        return output


class LRBatchNorm3d(BatchNorm3d):
    """Left-right reflection equivariant batch normalization 3D

    """
    def forward(self, input):
        n, _, x, y, z = input.shape
        input = input.view(n, -1, 2 * x, y, z)
        output = super().forward(input)
        output = output.view(n, -1, x, y, z)
        return output


class LRInstanceNorm2d(InstanceNorm2d):
    """Left-right reflection equivariant instance normalization 2D

    """
    def forward(self, input):
        n, _, x, y = input.shape
        input = input.view(n, -1, 2 * x, y)
        output = super().forward(input)
        output = output.view(n, -1, x, y)
        return output


class LRInstanceNorm3d(InstanceNorm3d):
    """Left-right reflection equivariant instance normalization 3D

    """
    def forward(self, input):
        n, _, x, y, z = input.shape
        input = input.view(n, -1, 2 * x, y, z)
        output = super().forward(input)
        output = output.view(n, -1, x, y, z)
        return output
        # self._check_input_dim(input)


        # # weight1 = interleave_bias(torch.ones_like(self.weight))
        # weight1 = interleave_bias(self.weight)
        # bias = interleave_bias(self.bias)
        # output = torch.nn.functional.instance_norm(
        #     input, self.running_mean, self.running_var, weight1, bias,
        #     self.training or not self.track_running_stats, self.momentum, self.eps)

        # import numpy as np
        # for i in range(input.shape[0]):
        #     print('sample', i)
        #     for j in range(input.shape[1]):
        #         print('channel', j)
        #         mean = np.mean(input[i, j, ...].detach().cpu().numpy())
        #         std = np.sqrt(np.var(input[i, j, ...].detach().cpu().numpy()) + self.eps)
        #         tmp = (input[i, j, ...] - mean) / std * self.weight[j//2] + self.bias[j//2]
        #         print('mean', mean, 'std', std)
        #         print(torch.max(torch.abs(tmp - output[i, j, ...])))
        # print('---')

        # # weight2 = interleave_bias(self.weight)
        # # bias = interleave_bias(self.bias)
        # # output *= weight.view(1, -1, 1, 1, 1)
        # return output
