# -*- coding: utf-8 -*-

from torch.nn.modules.loss import _WeightedLoss
from pytorch_engine.config import Config
from pytorch_engine.funcs import calc_dice_loss, prob_encode


class DiceLoss(_WeightedLoss):
    """Wrapper of Dice loss"""
    def __init__(self, weight=None):
        super().__init__(weight=weight)
    def forward(self, input, target):
        input = prob_encode(input)
        target_onehot = target.float()
        average = not Config().eval_separate
        return calc_dice_loss(input, target_onehot, weight=self.weight,
                              average=average)
