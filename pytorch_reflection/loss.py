# -*- coding: utf-8 -*-

from torch.nn.modules.loss import _WeightedLoss
from pytorch_metrics.funcs import calc_dice_loss, prob_encode


class DiceLoss(_WeightedLoss):
    """Wrapper of Dice loss.

    """
    def __init__(self, weight=None, average=True):
        super().__init__(weight=weight)
        self.average = average

    def forward(self, input, target):
        input = prob_encode(input)
        target_onehot = target.float()
        return calc_dice_loss(input, target_onehot, weight=self.weight,
                              average=self.average)
