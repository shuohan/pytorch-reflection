#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torch.nn.functional import relu

from pytorch_reflection.conv import LRConvI2F3d, LRConvF2F3d, LRConvF2I3d
from pytorch_reflection.norm import LRInstanceNorm3d, LRBatchNorm3d


Norm = LRBatchNorm3d
affine = True 

class Net1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = LRConvI2F3d(in_channels, 2, 3, padding=1, bias=False)
        self.norm1 = Norm(2, affine=affine)
        self.dp1 = torch.nn.Dropout3d(0.2)
        # self.norm1 = torch.nn.GroupNorm(2, 2 * 2, affine=True)
        self.conv2 = LRConvF2F3d(2, 4, 3, padding=1, bias=False)
        self.norm2 = Norm(4, affine=affine)
        # self.norm2 = torch.nn.GroupNorm(4, 2 * 4, affine=True)
        self.dp2 = torch.nn.Dropout3d(0.2)
        self.conv3 = LRConvF2F3d(4, 8, 3, padding=1, bias=False)
        self.norm3 = Norm(8, affine=affine)
        # self.norm3 = torch.nn.GroupNorm(8, 2 * 8, affine=True)
        self.dp3 = torch.nn.Dropout3d(0.2)
        self.conv4 = LRConvF2I3d(8, out_channels, 3, padding=1)

    def forward(self, input):
        output = self.conv1(input)
        output = self.norm1(output)
        output = relu(output)
        output = self.dp1(output)
        output = self.conv2(output)
        output = self.norm2(output)
        output = relu(output)
        output = self.dp2(output)
        output = self.conv3(output)
        output = self.norm3(output)
        output = relu(output)
        output = self.dp3(output)
        output = self.conv4(output)
        return output


class Net2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(in_channels, 4, 3, padding=1, bias=False)
        self.norm1 = torch.nn.BatchNorm3d(4)
        self.dp1 = torch.nn.Dropout3d(0.2)
        self.conv2 = torch.nn.Conv3d(4, 8, 3, padding=1, bias=False)
        self.norm2 = torch.nn.BatchNorm3d(8)
        self.dp2 = torch.nn.Dropout3d(0.2)
        self.conv3 = torch.nn.Conv3d(8, 16, 3, padding=1, bias=False)
        self.norm3 = torch.nn.BatchNorm3d(16)
        self.dp3 = torch.nn.Dropout3d(0.2)
        self.conv4 = torch.nn.Conv3d(16, out_channels, 3, padding=1)

    def forward(self, input):
        output = self.conv1(input)
        output = self.norm1(output)
        output = relu(output)
        output = self.dp1(output)
        output = self.conv2(output)
        output = self.norm2(output)
        output = relu(output)
        output = self.dp2(output)
        output = self.conv3(output)
        output = self.norm3(output)
        output = relu(output)
        output = self.dp3(output)
        output = self.conv4(output)
        return output


filename = 'image.nii.gz'
data = np.abs(nib.load(filename).get_data())
data = (data / 1000).astype(int)
data = torch.from_numpy(data).float()[None, None, ...].cuda()
# data = torch.arange(27).float().view(1, 1, 3, 3, 3).cuda()
flipped_data = data.flip(2)

net1 = Net1(1, 1).cuda().eval()
output1 = net1(data).detach().cpu().numpy()
output1_r = net1(flipped_data).detach().cpu().numpy()
diff1 = np.abs(output1[:, :, ::-1, ...] - output1_r)
diff1_tmp = diff1 / output1_r
diff1_tmp = diff1_tmp[np.logical_not(np.isnan(diff1_tmp))]

net2 = Net2(1, 1).cuda().eval()
output2 = net2(data).detach().cpu().numpy()
output2_r = net2(flipped_data).cpu().detach().numpy()
diff2 = np.abs(output2[:, :, ::-1, ...] - output2_r)
diff2_tmp = diff2 / output2_r
diff2_tmp = diff2_tmp[np.logical_not(np.isnan(diff2_tmp))]

slice_ind = data.shape[-1] // 2
data = data.cpu().numpy()
flipped_data = flipped_data.cpu().numpy()

plt.figure()
plt.subplot(3, 2, 1)
plt.imshow(data[0, 0, ..., slice_ind], cmap='gray')
plt.subplot(3, 2, 2)
plt.imshow(output1[0, 0, ..., slice_ind], cmap='gray')
plt.subplot(3, 2, 3)
plt.imshow(flipped_data[0, 0, ..., slice_ind], cmap='gray')
plt.subplot(3, 2, 4)
plt.imshow(output1_r[0, 0, ...,  slice_ind], cmap='gray')
plt.subplot(3, 2, 5)
plt.imshow(diff1[0, 0, ..., slice_ind])
text = 'max abs diff %.2e\nmax relative diff %.2e\noutput intensity [%.2e, %.2e]'
text = text %  (np.max(diff1), np.max(diff1_tmp), np.min(output1), np.max(output1))
plt.gcf().text(.5, .01, text, ha='center')

plt.figure()
plt.subplot(3, 2, 1)
plt.imshow(data[0, 0, ..., slice_ind], cmap='gray')
plt.subplot(3, 2, 2)
plt.imshow(output2[0, 0, ..., slice_ind], cmap='gray')
plt.subplot(3, 2, 3)
plt.imshow(flipped_data[0, 0, ..., slice_ind], cmap='gray')
plt.subplot(3, 2, 4)
plt.imshow(output2_r[0, 0, ...,  slice_ind], cmap='gray')
plt.subplot(3, 2, 5)
plt.imshow(diff2[0, 0, ..., slice_ind])
text = 'max abs diff %.2e\nmax relative diff %.2e\noutput intensity [%.2e, %.2e]'
text = text %  (np.max(diff2), np.max(diff2_tmp), np.min(output2), np.max(output2))
plt.gcf().text(.5, .01, text, ha='center')

plt.show()
