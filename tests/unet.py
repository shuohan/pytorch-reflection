#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torchviz import make_dot
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from image_processing_3d import padcrop3d
from pytorch_trainer.funcs import count_trainable_paras

from pytorch_reflection.unet import LRUNet, UNet, LRSegUNet


def eval_net(net, name):

    image = nib.load('image.nii.gz').get_data()
    # image = padcrop3d(image, (192, 256, 192))[0][None, None, ...]
    image = padcrop3d(image, (128, 96, 96))[0][None, None, ...]
    image = torch.from_numpy(image).float() # .cuda()
    flipped_image = image.flip(2)

    # print(net)
    print('# trainable parameters', count_trainable_paras(net))
    # output = net(image)
    # output_r = net(flipped_image)

    # dot = make_dot(output, params=dict(net.named_parameters()))
    # dot.format = 'svg'
    # dot.render(name)

    # output = output.detach().cpu().numpy()
    # output_r = output_r.detach().cpu().numpy()
    # image = image.detach().cpu().numpy()
    # flipped_image = flipped_image.detach().cpu().numpy()

    # diff = np.abs(output[:, :, ::-1, ...] - output_r)
    # diff_r = diff / output_r
    # diff_r = diff_r[np.logical_not(np.isnan(diff_r))]

    # slice_ind = image.shape[-1] // 2
    # plt.figure()
    # plt.subplot(3, 2, 1)
    # plt.imshow(image[0, 0, ..., slice_ind], cmap='gray')
    # plt.subplot(3, 2, 2)
    # plt.imshow(output[0, 0, ..., slice_ind], cmap='gray')
    # plt.subplot(3, 2, 3)
    # plt.imshow(flipped_image[0, 0, ..., slice_ind], cmap='gray')
    # plt.subplot(3, 2, 4)
    # plt.imshow(output_r[0, 0, ...,  slice_ind], cmap='gray')
    # plt.subplot(3, 2, 5)
    # plt.imshow(diff[0, 0, ..., slice_ind])
    # text = 'max abs diff %.2e\nmax relative diff %.2e\noutput intensity [%.2e, %.2e]'
    # text = text %  (np.max(diff), np.max(diff_r), np.min(output), np.max(output))
    # plt.gcf().text(.5, .01, text, ha='center')


# first_channels = 24
num_trans_down = 5
# out_classes = 18
# net = LRUNet(1, out_classes, num_trans_down, first_channels) #.cuda().eval()
# eval_net(net, 'lr_unet')

first_channels = 31
out_classes = 29
net = UNet(1, out_classes, num_trans_down, first_channels) # .cuda().eval()
eval_net(net, 'unet')

# plt.show()
