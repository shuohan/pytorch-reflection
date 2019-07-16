#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torch.nn.functional import relu

from pytorch_reflection.conv import LRConvI2F2d, LRConvF2F2d, LRConvF2I2d
from pytorch_reflection.norm import LRInstanceNorm2d, LRBatchNorm2d


class Net1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = LRConvI2F2d(in_channels, 2, 3, padding=1, use_bias=False)
        # self.norm1 = LRInstanceNorm2d(2, affine=True)
        self.norm1 = LRBatchNorm2d(2, affine=True)
        self.conv2 = LRConvF2F2d(2, 4, 3, padding=1, use_bias=False)
        # self.norm2 = LRInstanceNorm2d(4, affine=True)
        self.norm2 = LRBatchNorm2d(4, affine=True)
        self.conv3 = LRConvF2F2d(4, 8, 3, padding=1, use_bias=False)
        # self.norm3 = LRInstanceNorm2d(8, affine=True)
        self.norm3 = LRBatchNorm2d(8, affine=True)
        self.conv4 = LRConvF2I2d(8, out_channels, 3, padding=1)

    def forward(self, input):
        output = self.conv1(input)
        output = self.norm1(output)
        output = relu(output)
        output = self.conv2(output)
        output = self.norm2(output)
        output = relu(output)
        output = self.conv3(output)
        output = self.norm3(output)
        output = relu(output)
        output = self.conv4(output)
        print(self.norm3.running_mean)
        return output


class Net2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 4, 3, padding=1, bias=False)
        self.norm1 = torch.nn.BatchNorm2d(4)
        self.conv2 = torch.nn.Conv2d(4, 8, 3, padding=1, bias=False)
        self.norm2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 16, 3, padding=1, stride=2, bias=False)
        self.norm3 = torch.nn.BatchNorm2d(16)
        self.conv4 = torch.nn.Conv2d(16, out_channels, 3, padding=1)

    def forward(self, input):
        output = self.conv1(input)
        output = self.norm1(output)
        output = relu(output)
        output = self.conv2(output)
        output = self.norm2(output)
        output = relu(output)
        output = self.conv3(output)
        output = self.norm3(output)
        output = relu(output)
        output = self.conv4(output)
        return output


filename = 'image.nii.gz'
data = nib.load(filename).get_data()
slice_ind1 = 100
slice_ind2 = 125
data1 = torch.from_numpy(data[..., slice_ind1]).float()[None, None, ...]
data2 = torch.from_numpy(data[..., slice_ind2]).float()[None, None, ...]
data = torch.cat((data1, data2), 0)
flipped_data = data.flip(2)

net1 = Net1(1, 1)
output1 = net1(data).detach().numpy()
output1_r = net1(flipped_data).detach().numpy()

net1 = net1.eval()
output1 = net1(data).detach().numpy()
output1_r = net1(flipped_data).detach().numpy()

diff1 = np.abs(output1[0, 0, ::-1, ...] - output1_r[0, 0, ...])

plt.figure()
plt.subplot(3, 2, 1)
plt.imshow(data[0, ...].squeeze(), cmap='gray')
plt.subplot(3, 2, 2)
plt.imshow(output1[0, ...].squeeze(), cmap='gray')
plt.subplot(3, 2, 3)
plt.imshow(flipped_data[0, ...].squeeze(), cmap='gray')
plt.subplot(3, 2, 4)
plt.imshow(output1_r[0, ...].squeeze(), cmap='gray')
plt.subplot(3, 2, 5)
plt.imshow(diff1)
text = 'max abs diff %.2e\noutput intensity [%.2e, %.2e]'
text = text %  (np.max(diff1), np.min(output1), np.max(output1))
plt.gcf().text(.5, .01, text, ha='center')

net2 = Net2(1, 1)
output2 = net2(data).detach().numpy()
output2_r = net2(flipped_data).detach().numpy()
diff2 = np.abs(output2[0, 0, ::-1, ...] - output2_r[0, 0, ...])

plt.figure()
plt.subplot(3, 2, 1)
plt.imshow(data[0, ...].squeeze(), cmap='gray')
plt.subplot(3, 2, 2)
plt.imshow(output2[0, ...].squeeze(), cmap='gray')
plt.subplot(3, 2, 3)
plt.imshow(flipped_data[0, ...].squeeze(), cmap='gray')
plt.subplot(3, 2, 4)
plt.imshow(output2_r[0, ...].squeeze(), cmap='gray')
plt.subplot(3, 2, 5)
plt.imshow(diff2)
text = 'max abs diff %.2e\noutput intensity [%.2e, %.2e]'
text = text %  (np.max(diff2), np.min(output2), np.max(output2))
plt.gcf().text(.5, .01, text, ha='center')

plt.show()
