# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np


class _LRConvI2F(torch.nn.modules.conv._ConvNd):
    """Left-right reflection equivariant convolution from image to feature maps.

    The 0 dimension is assumed to be left and right. It can take a
    multi-channel image with shape (num_batch, in_channels, *spatial_size) to
    output feature maps with shape (num_batch, out_channels * 2, *spatial_size).

    Attributes:
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels in the output image.
        kernel_size (tuple): The :class:`int` size of convolving kernel.
        stride (tuple, optional): The :class:`int` stride of the convolution.
        padding (tuple, optional):
        padding_mode (str, optional):
        bias (bool): Add a learnable bias to the output if ``True``.

    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, padding_mode='zeros', use_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.use_bias = use_bias

        weight = torch.Tensor(out_channels, in_channels, *kernel_size)
        self.weight = torch.nn.Parameter(weight)
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def forward(self, input):
        indices = np.arange(self.kernel_size[0] - 1, -1, -1)
        flipped_weight = self.weight[:, :, indices, ...]
        weight = torch.cat((self.weight, flipped_wight), 0)
        output = self._conv(input, weight, self.stride, self.padding)
        if self.bias is not None:
            bias = self.bias.repeat(2)
            bias = bias.view(1, -1, *np.ones(len(self.kernel_size, dtype=int)))
            output = output + bias
        return output

    def _conv(self, input, weight, stride, padding):
        raise NotImplementedError


class LRConvI2F2d(_LRConvI2F):
    """2D left-right reflection equivariant conv from image to feature maps.

    Attributes:
        kernel_size (int or tuple):
        stride (int or tuple):
        padding (int or tuple):

    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, padding_mode='zeros', use_bias=True):
        kernel_size = torch.nn.modules.utils._pair(kernel_size)
        stride = torch.nn.modules.utils._pair(kernel_size)
        padding = torch.nn.modules.utils._pair(kernel_size)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, padding_mode, use_bias)

    def _conv(self, in_channels, weight, stride, padding):
        return F.conv2d(in_channels, weight, stride=stride, padding=padding)


class LRConvI2F3d(_LRConvI2F):
    """3D left-right reflection equivariant conv from image to feature maps.

    Attributes:
        kernel_size (int or tuple):
        stride (int or tuple):
        padding (int or tuple):

    """
    def __init__(self, in_channels, out_channels, kernel_size):
        kernel_size = torch.nn.modules.utils._triple(kernel_size)
        stride = torch.nn.modules.utils._triple(kernel_size)
        padding = torch.nn.modules.utils._triple(kernel_size)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, padding_mode, use_bias)

    def _conv(self, in_channels, weight, stride, padding):
        return F.conv3d(in_channels, weight, stride=stride, padding=padding)


class _LRConvF2F(torch.nn.modules.conv._ConvNd):
    """Left-right reflection equivariant convolution from features to features.

    The 0 dimension is assumed to be left and right. It can take input
    feature maps with shape (num_batch, in_channels * 2, *spatial_size) to
    output feature maps with shape (num_batch, out_channels * 2, *spatial_size).

    Note:
        Use num_labels / 2 as ``out_channels`` for symmetric
        segmentation/parcellation.
    
    Attributes:
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels in the output image.
        kernel_size (tuple): The :class:`int` size of convolving kernel.
        stride (tuple, optional): The :class:`int` stride of the convolution.
        padding (tuple, optional):
        padding_mode (str, optional):
        bias (bool): Add a learnable bias to the output if ``True``.

    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, padding_mode='zeros', use_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.use_bias = use_bias

        weight1 = torch.Tensor(out_channels, in_channels, *kernel_size)
        self.weight1 = torch.nn.Parameter(weight1)
        weight2 = torch.Tensor(out_channels, in_channels, *kernel_size)
        self.weight2 = torch.nn.Parameter(weight2)
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def forward(self, input):
        indices = np.arange(self.kernel_size[0] - 1, -1, -1)
        flipped_weight1 = self.weight1[:, :, indices, ...]
        flipped_weight2 = self.weight2[:, :, indices, ...]
        weight1 = torch.cat((self.weight1, self.weight2), 0)
        weight2 = torch.cat((flipped_weight2, flipped_weight1), 0)
        output1 = self._conv(input, weight1, self.stride, self.padding)
        output2 = self._conv(input, weight1, self.stride, self.padding)
        if self.bias is not None:
            bias = self.bias.repeat(2)
            bias = bias.view(1, -1, *np.ones(len(self.kernel_size, dtype=int)))
            output1 = output1 + bias
            output2 = output2 + bias
        output = torch.cat((output1, output2), 1)
        return output

    def _conv(self, input, weight, stride, padding):
        raise NotImplementedError


class LRConvF2F2d(_LRConvF2F):
    """2D left-right reflection equivariant conv from features to features.

    Attributes:
        kernel_size (int or tuple):
        stride (int or tuple):
        padding (int or tuple):

    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, padding_mode='zeros', use_bias=True):
        kernel_size = torch.nn.modules.utils._pair(kernel_size)
        stride = torch.nn.modules.utils._pair(kernel_size)
        padding = torch.nn.modules.utils._pair(kernel_size)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, padding_mode, use_bias)

    def _conv(self, in_channels, weight, stride, padding):
        return F.conv2d(in_channels, weight, stride=stride, padding=padding)


class LRConvF2F3d(_LRConvF2F):
    """3D left-right reflection equivariant conv from features to features.

    Attributes:
        kernel_size (int or tuple):
        stride (int or tuple):
        padding (int or tuple):

    """
    def __init__(self, in_channels, out_channels, kernel_size):
        kernel_size = torch.nn.modules.utils._triple(kernel_size)
        stride = torch.nn.modules.utils._triple(kernel_size)
        padding = torch.nn.modules.utils._triple(kernel_size)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, padding_mode, use_bias)

    def _conv(self, in_channels, weight, stride, padding):
        return F.conv3d(in_channels, weight, stride=stride, padding=padding)


class _LRConvF2I(torch.nn.modules.conv._ConvNd):
    """Left-right reflection equivariant convolution from features to image.

    The 0 dimension is assumed to be left and right. It can take feature maps
    with shape (num_batch, in_channels * 2, *spatial_size) to output
    multi-channel image with shape (num_batch, out_channels, *spatial_size).

    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, padding_mode='zeros', use_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.use_bias = use_bias

        weight1 = torch.Tensor(out_channels, in_channels, *kernel_size)
        self.weight1 = torch.nn.Parameter(weight1)
        weight2 = torch.Tensor(out_channels, in_channels, *kernel_size)
        self.weight2 = torch.nn.Parameter(weight2)
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def forward(self, input):
        indices = np.arange(self.kernel_size[0] - 1, -1, -1)
        flipped_weight1 = self.weight1[:, :, indices, ...]
        flipped_weight2 = self.weight2[:, :, indices, ...]
        weight1 = torch.cat((self.weight1, self.weight2), 0)
        weight2 = torch.cat((flipped_weight2, flipped_weight1), 0)
        output1 = self._conv(input, weight1, self.stride, self.padding)
        output2 = self._conv(input, weight1, self.stride, self.padding)
        if self.bias is not None:
            bias = self.bias.repeat(2)
            bias = bias.view(1, -1, *np.ones(len(self.kernel_size, dtype=int)))
            output1 = output1 + bias
            output2 = output2 + bias
        output = output1 + output2
        return output


class LRConvF2I2d(_LRConvF2I):
    """2D left-right reflection equivariant conv from features to image.

    Attributes:
        kernel_size (int or tuple):
        stride (int or tuple):
        padding (int or tuple):

    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, padding_mode='zeros', use_bias=True):
        kernel_size = torch.nn.modules.utils._pair(kernel_size)
        stride = torch.nn.modules.utils._pair(kernel_size)
        padding = torch.nn.modules.utils._pair(kernel_size)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, padding_mode, use_bias)

    def _conv(self, in_channels, weight, stride, padding):
        return F.conv2d(in_channels, weight, stride=stride, padding=padding)


class LRConvF2I3d(_LRConvF2I):
    """3D left-right reflection equivariant conv from features to image.

    Attributes:
        kernel_size (int or tuple):
        stride (int or tuple):
        padding (int or tuple):

    """
    def __init__(self, in_channels, out_channels, kernel_size):
        kernel_size = torch.nn.modules.utils._triple(kernel_size)
        stride = torch.nn.modules.utils._triple(kernel_size)
        padding = torch.nn.modules.utils._triple(kernel_size)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, padding_mode, use_bias)

    def _conv(self, in_channels, weight, stride, padding):
        return F.conv3d(in_channels, weight, stride=stride, padding=padding)
