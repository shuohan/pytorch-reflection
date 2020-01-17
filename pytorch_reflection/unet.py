# -*- coding: utf-8 -*-
  
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_layers import create_two_avg_pool, create_k1_conv
from pytorch_layers import Config as LConfig
from pytorch_layers import Dim

from .blocks import LRInputBlock, LRContractingBlock, LRExpandingBlock
from .blocks import create_LRF2I, LRTransUpBlock, TransUpBlock
from .blocks import InputBlock, ContractingBlock, ExpandingBlock
from .blocks import create_LRF2F_proj


class _UNet(torch.nn.Module):
    def __init__(self, in_channels, out_classes, num_trans_down,
                 first_channels, max_channels=1024):
        super().__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.num_trans_down = num_trans_down
        self.max_channels = max_channels

        # encoding/contracting
        inter_channels = (in_channels + first_channels) // 2
        self.cb0 = self._create_ib(in_channels, first_channels, inter_channels)
        in_channels = first_channels
        for i in range(self.num_trans_down):
            out_channels = self._calc_out_channels(in_channels)
            inter_channels = (in_channels + out_channels) // 2
            setattr(self, 'td%d'%(i), create_two_avg_pool())
            cb = self._create_cb(in_channels, out_channels, inter_channels)
            setattr(self, 'cb%d'%(i+1), cb)
            in_channels = out_channels

        # decoding/expanding
        for i in range(self.num_trans_down):
            shortcut_ind = self.num_trans_down - i - 1
            out_channels = getattr(self, 'cb%d'%shortcut_ind).out_channels
            setattr(self, 'tu%d'%i, self._create_tu(in_channels, out_channels))
            eb = self._create_eb(out_channels, out_channels, out_channels)
            setattr(self, 'eb%d'%i, eb)
            in_channels = out_channels

        # output
        self.out = self._create_out(out_channels)


    def _calc_out_channels(self, in_channels):
        out_channels = min(in_channels * 2, self.max_channels)
        return out_channels

    def _create_ib(self, in_channels, out_channels, inter_channels):
        """Returns an input block"""
        raise NotImplementedError

    def _create_cb(self, in_channels, out_channels, inter_channels):
        """Returns a contracting block"""
        raise NotImplementedError

    def _create_tu(self, in_channels, out_channels):
        """Returns a trans up block"""
        raise NotImplementedError

    def _create_eb(self, in_channels, shortcut_channels, out_channels):
        """Returns a contracting block"""
        raise NotImplementedError

    def _create_out(self, in_channels):
        """Returns an output layler"""
        raise NotImplementedError

    def forward(self, input):
        # encoding/contracting
        output = input
        shortcuts = list()
        for i in range(self.num_trans_down+1):
            output = getattr(self, 'cb%d'%i)(output)
            if i < self.num_trans_down:
                shortcuts.insert(0, output)
                output = getattr(self, 'td%d'%(i))(output)

        # decoding/expanding
        for i, shortcut in enumerate(shortcuts):
            output = getattr(self, 'tu%d'%i)(output)
            output = getattr(self, 'eb%d'%i)(output, shortcut)

        output = self.out(output)
        return output


class LRUNet(_UNet):
    def _create_ib(self, in_channels, out_channels, inter_channels):
        return LRInputBlock(in_channels, out_channels, inter_channels)
    
    def _create_cb(self, in_channels, out_channels, inter_channels):
        return LRContractingBlock(in_channels, out_channels, inter_channels)

    def _create_tu(self, in_channels, out_channels):
        return LRTransUpBlock(in_channels, out_channels)

    def _create_eb(self, in_channels, shortcut_channels, out_channels):
        return LRExpandingBlock(in_channels, shortcut_channels, out_channels)

    def _create_out(self, in_channels):
        return create_LRF2I(in_channels, self.out_classes)


class LRSegOut(torch.nn.Module):
    def __init__(self, in_channels, out_channels, paired_labels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.paired_labels = paired_labels
        self.single_labels = self._calc_single_labels()
        print(self.paired_labels, self.single_labels)
        self.conv = self._create_conv()
        self.weight = torch.nn.Parameter(self._construct_weight())
        self.weight.requires_grad = False
        # torch.set_printoptions(threshold=5000)
        # print(self.weight.squeeze().int().data)
        # print(torch.sum(self.weight.data, 0).squeeze().int().data)

    def _calc_single_labels(self):
        all_labels = set(np.arange(self.out_channels))
        paired_labels = set(np.array(self.paired_labels).flatten())
        return sorted(list(all_labels - paired_labels))

    def _create_conv(self):
        out_channels = (self.out_channels + len(self.single_labels)) // 2
        return create_LRF2F_proj(self.in_channels, out_channels)

    def _construct_weight(self):
        in_channels = self.conv.out_channels * 2
        tmp_weights = dict()
        counter = 0
        for label in self.single_labels:
            weight = torch.zeros(1, in_channels)
            weight[0, counter] = 1
            weight[0, counter + 1] = 1
            tmp_weights[label] = weight
            counter = counter + 2
        for label1, label2 in self.paired_labels:
            weight1 = torch.zeros(1, in_channels)
            weight2 = torch.zeros(1, in_channels)
            weight1[0, counter] = 1
            weight2[0, counter + 1] = 1
            tmp_weights[label1] = weight1
            tmp_weights[label2] = weight2
            counter = counter + 2
        weights = list()
        for key in sorted(list(tmp_weights.keys())):
            weights.append(tmp_weights[key])
        weights = torch.cat(weights, dim=0)
        if LConfig.dim is Dim.TWO:
            return weights[..., None, None]
        elif LConfig.dim is Dim.THREE:
            return weights[..., None, None, None]

    def extra_repr(self):
        s = 'single={single_labels}, paired={paired_labels}'
        return s.format(**self.__dict__)

    def forward(self, input):
        output = self.conv(input)
        print(self.weight.requires_grad)
        if LConfig.dim is Dim.TWO:
            return F.conv2d(output, self.weight)
        elif LConfig.dim is Dim.THREE:
            return F.conv3d(output, self.weight)


class LRSegUNet(LRUNet):

    def __init__(self, in_channels, out_classes, num_trans_down,
                 first_channels, paired_labels, max_channels=1024):
        self.paired_labels = paired_labels
        super().__init__(in_channels, out_classes, num_trans_down,
                         first_channels, max_channels)

    def _create_out(self, in_channels):
        return LRSegOut(in_channels, self.out_classes, self.paired_labels)


class UNet(_UNet):
    def _create_ib(self, in_channels, out_channels, inter_channels):
        return InputBlock(in_channels, out_channels, inter_channels)
    
    def _create_cb(self, in_channels, out_channels, inter_channels):
        return ContractingBlock(in_channels, out_channels, inter_channels)

    def _create_tu(self, in_channels, out_channels):
        return TransUpBlock(in_channels, out_channels)

    def _create_eb(self, in_channels, shortcut_channels, out_channels):
        return ExpandingBlock(in_channels, shortcut_channels, out_channels)

    def _create_out(self, in_channels):
        return create_k1_conv(in_channels, self.out_classes)
