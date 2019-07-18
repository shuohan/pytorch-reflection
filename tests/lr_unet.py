#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torchviz import make_dot
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from image_processing_3d import padcrop3d

from pytorch_reflection.unet import LRUNet


first_channels = 2
num_trans_down = 4
out_classes = 1
# x, y, z = (64, 48, 48)

# image = torch.randn(1, 1, x, y, z)
image = nib.load('image.nii.gz').get_data()
image = padcrop3d(image, (192, 256, 192))[0][None, None, ...]
image = torch.from_numpy(image).float().cuda()
flipped_image = image.flip(2)

net = LRUNet(1, out_classes, num_trans_down, first_channels).cuda().eval()
print(net)
output = net(image)
output_r = net(flipped_image)

dot = make_dot(output, params=dict(net.named_parameters()))
dot.format = 'svg'
dot.render('lr_unet')

output = output.detach().cpu().numpy()
output_r = output_r.detach().cpu().numpy()
image = image.detach().cpu().numpy()
flipped_image = flipped_image.detach().cpu().numpy()

diff = np.abs(output[:, :, ::-1, ...] - output_r)
diff_r = diff / output_r
diff_r = diff_r[np.logical_not(np.isnan(diff_r))]

slice_ind = image.shape[-1] // 2
plt.figure()
plt.subplot(3, 2, 1)
plt.imshow(image[0, 0, ..., slice_ind], cmap='gray')
plt.subplot(3, 2, 2)
plt.imshow(output[0, 0, ..., slice_ind], cmap='gray')
plt.subplot(3, 2, 3)
plt.imshow(flipped_image[0, 0, ..., slice_ind], cmap='gray')
plt.subplot(3, 2, 4)
plt.imshow(output_r[0, 0, ...,  slice_ind], cmap='gray')
plt.subplot(3, 2, 5)
plt.imshow(diff[0, 0, ..., slice_ind])
text = 'max abs diff %.2e\nmax relative diff %.2e\noutput intensity [%.2e, %.2e]'
text = text %  (np.max(diff), np.max(diff_r), np.min(output), np.max(output))
plt.gcf().text(.5, .01, text, ha='center')
plt.show()

