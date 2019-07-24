#!/usr/bin/env python
# -*- coding: utf-8 -*-
  
import numpy as np
from glob import glob
import os


sub_ids = list()
for image in sorted(glob('data/*image.nii.gz')):
    sub_ids.append(os.path.basename(image).replace('_image.nii.gz', ''))

num_sel = len(sub_ids) // 2
tra_ids = np.random.choice(sub_ids, num_sel, replace=False)
val_ids = list(set(sub_ids) - set(tra_ids))


def link(target_dirname, ids):
    if not os.path.isdir(target_dirname):
        os.makedirs(target_dirname)

    for i in ids:
        image = os.path.join('data', i + '_image.nii.gz')
        mask = os.path.join('data', i + '_mask.nii.gz')
        target_image = os.path.join(target_dirname, os.path.basename(image))
        target_mask = os.path.join(target_dirname, os.path.basename(mask))
        source_image = os.path.join('..', image)
        source_mask = os.path.join('..', mask)
        os.symlink(source_image, target_image)
        os.symlink(source_mask, target_mask)
    labels = '../data/labels.json'
    os.symlink(labels, os.path.join(target_dirname, 'labels.json'))

link('training_data', tra_ids)
link('validation_data', val_ids)

# check
tra = set([os.path.basename(fn) for fn in glob('training_data/*image.nii.gz')])
val = set([os.path.basename(fn) for fn in glob('validation_data/*image.nii.gz')])
assert len(tra | val) == 125

tra = set([os.path.basename(fn) for fn in glob('training_data/*mask.nii.gz')])
val = set([os.path.basename(fn) for fn in glob('validation_data/*mask.nii.gz')])
assert len(tra | val) == 125
