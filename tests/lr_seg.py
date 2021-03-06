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
from dataset import DatasetCreator, Config


def eval_net(net, name):

    chid = 3

    image = nib.load('../scripts/training_ma/1000_image.nii.gz').get_data()
    # image = padcrop3d(image, (192, 256, 192))[0][None, None, ...]
    # image = padcrop3d(image, (192, 256, 192))[0][None, None, ...]
    image = padcrop3d(image, (96, 96, 96))[0][None, None, ...]
    # image = padcrop3d(image, (96, 96, 96))[0][None, None, ...]
    image = torch.from_numpy(image.astype(np.float32)).cuda()
    flipped_image = image.flip(2)

    print(net)
    print('# trainable parameters', count_trainable_paras(net))
    output = net(image)
    output_r = net(flipped_image)

    # dot = make_dot(output, params=dict(net.named_parameters()))
    # dot.format = 'svg'
    # dot.render(name)

    output = output.detach().cpu().numpy()
    output_r = output_r.detach().cpu().numpy()
    image = image.detach().cpu().numpy()
    flipped_image = flipped_image.detach().cpu().numpy()

    print(output.shape)
    print(output_r.shape)

    single_labels = net.out.single_labels
    paired_labels = net.out.paired_labels
    single_diff = np.abs(output[:, single_labels, ::-1, ...] \
                         - output_r[:, single_labels, ...])
    paired_diff = list()
    for l1, l2 in paired_labels:
        diff1 = np.abs(output[:, l1:(l1+1), ::-1, ...] - output_r[:, l2:(l2+1), ...])
        diff2 = np.abs(output[:, l2:(l2+1), ::-1, ...] - output_r[:, l1:(l1+1), ...])
        paired_diff.append(diff1)
        paired_diff.append(diff2)
    paired_diff = np.concatenate(paired_diff, axis=1)
    diff = np.concatenate((single_diff, paired_diff), axis=1)
    print(diff.shape)
    diff_r = diff / output_r
    diff_r[np.isnan(diff_r)] = 0
    diff = np.abs(diff)
    diff_r = np.abs(diff_r)

    slice_ind = image.shape[-1] // 2
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.imshow(image[0, 0, ..., slice_ind], cmap='gray')
    plt.subplot(3, 2, 2)
    plt.imshow(output[0, chid, ..., slice_ind], cmap='gray')

    plt.subplot(3, 2, 3)
    plt.imshow(flipped_image[0, 0, ..., slice_ind], cmap='gray')
    plt.subplot(3, 2, 4)
    plt.imshow(output_r[0, chid, ...,  slice_ind], cmap='gray')

    plt.subplot(3, 2, 5)
    plt.imshow(diff[0, chid, ..., slice_ind])
    text = 'max abs diff %.2e\nmax relative diff %.2e\noutput intensity [%.2e, %.2e]'
    text = text %  (np.max(diff), np.max(diff_r), np.min(output), np.max(output))
    plt.gcf().text(.5, .01, text, ha='center')

    plt.subplot(3, 2, 6)
    plt.imshow(diff_r[0, chid, ..., slice_ind])


Config.label_suffixes = ['subcortical']
first_channels = 24
num_trans_down = 5
creator = DatasetCreator()
creator.add_image_type('label')
creator.add_dataset(dirname='../scripts/training_ma')
td = creator.create().dataset
label_image = td.images[list(td.images.keys())[0]][0].normalize()
labels = label_image.label_info.labels
paired_labels = label_image.label_info.pairs
print(label_image.label_info)
out_classes = len(labels)
print(out_classes)

net = LRSegUNet(1, out_classes, num_trans_down, first_channels,
                paired_labels).cuda().eval()
cp = torch.load('../scripts/1116_sub/checkpoint_800.pt')
net.load_state_dict(cp['model'])

eval_net(net, 'lr_seg_unet')
# 
plt.show()
