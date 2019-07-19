# -*- coding: utf-8 -*-
  
import torch
from pytorch_engine.layers import create_pool, create_proj

from .blocks import LRInputBlock, LRContractingBlock, LRExpandingBlock
from .blocks import create_LRF2I, LRTransUpBlock, TransUpBlock
from .blocks import InputBlock, ContractingBlock, ExpandingBlock


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
            setattr(self, 'td%d'%(i), create_pool(2))
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
        return create_proj(in_channels, self.out_classes)
