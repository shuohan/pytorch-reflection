#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description='Train hierarchical network')

help = ('Configuration .json file. If it is specified and exists, the '
        'command line arguments are ignored.')
parser.add_argument('-j', '--json-config', help=help, default='')

help = ('The data directory. *image.nii are the training images, *label.nii '
        'are the training truth label images, and *mask.nii are ROI masks')
parser.add_argument('-o', '--output-prefix', default='',
                    help='Outut model prefix; add slash (/) for folder')
parser.add_argument('-a', '--augmentation', nargs='+', default=list(),
                    choices={'rotate', 'deform', 'scale'},
                    help='Data augmentation methods; orders are preserved')
parser.add_argument('-f', '--flip', default=False, action='store_true',
                    help='Flip the data to augment.')
parser.add_argument('-tb', '--training-batch-size', type=int, default=1,
                    help='The number of images per training mini-batch')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=1,
                    help='The number of images per validation mini-batch')
parser.add_argument('-c', '--cropping-shape', nargs=3, type=int,
                    default=None, help='The shape of the cropped out region')
parser.add_argument('-s', '--input-shape', nargs=3, type=int,
                    default=(192, 256, 192),
                    help='The shape of the input to crop to.')
parser.add_argument('-e', '--num-epochs', type=int, default=200,
                    help='The number of epochs')
parser.add_argument('-mp', '--model-period', type=int, default=10,
                    help='Save the model every this number of epochs')
parser.add_argument('-n', '--num-workers', type=int, default=1,
                    help='Number of data loader workers')
parser.add_argument('-d', '--depth', type=int, default=4,
                    help='Number of contracting blocks')
parser.add_argument('-w', '--width', type=int, default=2,
                    help='Number of features of the first block output')
parser.add_argument('-m', '--checkpoint', default='',
                    help='The checkpoint to continue training')

parser.add_argument('-A', '--augmentation-prob', type=float, default=0.5,
                    help='The probability of applying augmentation')
parser.add_argument('-V', '--verbose', type=int, default=1,
                    help='Print dataset info')
parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                    help='Adam learning rate')
parser.add_argument('-td', '--training-dir', default=None,
                    help='Training data directories')
parser.add_argument('-vd', '--validation-dir', default=None,
                    help='Validation data directories')
parser.add_argument('-t', '--network-type', choices={'normal', 'lr', 'lr-seg'},
                    default='lr', help='The type of the network.')
parser.add_argument('-is', '--image-suffix', default='image',
                    help='The suffix of the image file.')
parser.add_argument('-ls', '--label-suffix', default='label',
                    help='The suffix of the label image file.')
parser.add_argument('-ms', '--mask-suffix', default='mask',
                    help='The suffix of the mask image file.')
parser.add_argument('-vp', '--val-period', type=int, default=1,
                    help='The epoch period of validation')
parser.add_argument('-pp', '--pred-period', type=int, default=1,
                    help='Save the prediction every this number of epochs')
parser.add_argument('-ss', '--separate-samples', default=False,
                    action='store_true', help='Print loss separately')
parser.add_argument('-do', '--dropout', default=0.2, type=float)

args = parser.parse_args()


import os
import json
import torch
from glob import glob
from torch.optim import Adam
from torch.utils.data import DataLoader

from pytorch_layers import Config as LConfig

from dataset import Config as DConfig
from dataset.datasets import DatasetCreator
from dataset.pipelines import RandomPipeline
from dataset.workers import WorkerCreator

from pytorch_trainer.config import Config as TConfig
from pytorch_trainer.config import LoggerFormat
from pytorch_trainer.tra_val import BasicTrainer, BasicValidator
from pytorch_trainer.loggers import Logger
from pytorch_trainer.printers import Printer
from pytorch_trainer.savers import ModelSaver, SegPredSaver
from pytorch_trainer.funcs import count_trainable_paras

from pytorch_reflection.worker import LabelReorderer
from pytorch_metrics import DiceLoss


# set configurations
if os.path.isfile(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    # script_config = checkpoint['script_config']
    # script_config['checkpoint'] = args.checkpoint
    # script_config['training_batch_size'] = args.training_batch_size
    # script_config['validation_batch_size'] = args.validation_batch_size
    # script_config['num_workers'] = args.num_workers
    # script_config['training_dir'] = args.training_dir
    # script_config['validation_dir'] = args.validation_dir
    # script_config['output_prefix'] = args.output_prefix
    # for key, value in script_config.items():
    #     setattr(args, key, value)

    print(checkpoint.keys())
    TConfig.load_dict(checkpoint['tainer_config'])
    TConfig.num_epochs = args.num_epochs
    TConfig.model_period = args.model_period
    TConfig.pred_period = args.pred_period
    TConfig.val_period = args.val_period

    # DConfig.load_dict(checkpoint['dataset_config'])
    # LConfig.load_dict(checkpoint['layers_config'])
else:
    script_config = args.__dict__

print('Script config')
# keylen = max([len(key)+1 for key in script_config.keys()])
# for key, value in script_config.items():
#     print('    %s %s' % ((key+':').ljust(keylen), value))

DConfig.verbose = args.verbose
DConfig.aug_prob = args.augmentation_prob
DConfig.crop_shape = args.cropping_shape
DConfig.image_shape = args.input_shape
DConfig.image_suffixes = [args.image_suffix]
DConfig.label_suffixes = [args.label_suffix]
DConfig.mask_suffixes = [args.mask_suffix]
print('Dataset config')
DConfig.show()
print('-----------')

LConfig.dropout = args.dropout
print('Layers config')
LConfig.show()
print('-----------')

TConfig.model_period = args.model_period
TConfig.pred_period = args.pred_period
TConfig.val_period = args.val_period
TConfig.num_epochs = args.num_epochs
TConfig.logger_fmt = LoggerFormat.LONG
print('Trainer config')
TConfig.show()
print('-----------')

# load datasets

def load_dataset(dirname, augmentation=[], flip=False):
    creator = DatasetCreator()
    creator.add_image_type('image', 'label')
    if args.cropping_shape is not None:
        creator.add_image_type('mask')
    else:
        creator.add_operation('resize')

    creator.add_operation(*augmentation)
    if args.cropping_shape is not None:
        creator.add_operation('crop')
    creator.add_operation('norm_label')
    creator.add_dataset(dirname)
    dataset = creator.create().dataset

    if flip:
        pipeline = RandomPipeline()
        pipeline.register('flip')
        if args.cropping_shape is None:
            pipeline.register('resize')
        pipeline.register(*augmentation)
        if args.cropping_shape is not None:
            pipeline.register('crop')
        pipeline.register('norm_label')
        dataset.add_pipeline(pipeline)

    return dataset

td = load_dataset(args.training_dir, args.augmentation, args.flip)
vd = load_dataset(args.validation_dir, flip=True)

# print datasets
print('-' * 80)
print('training dataset')
print('# training data', len(td))
print(td)
print('-' * 80)
print('validation dataset')
print('# validation data', len(vd))
print(vd)

out_classes = len(list(td.labels.keys())[0].labels)
if out_classes == 2:
    out_classes -= 1

if args.network_type == 'lr':
    from pytorch_reflection.unet import LRUNet as Net
elif args.network_type == 'normal':
    from pytorch_reflection.unet import UNet as Net
else:
    from pytorch_reflection.unet import LRSegUNet as Net


if args.network_type == 'lr-seg':
    label_image = td.images[list(td.images.keys())[0]][1].normalize()
    paired_labels = label_image.label_info.pairs
    print(label_image.label_info)
    net = Net(1, out_classes, args.depth, args.width, paired_labels).cuda()
else:
    net = Net(1, out_classes, args.depth, args.width).cuda()

if os.path.isfile(args.checkpoint):
    args.output_prefix += 'cont_'
    net.load_state_dict(checkpoint['model'])

# print model
print(net)
print('-' * 80)
print('#paras:', count_trainable_paras(net))
print('-' * 80)

# loss and optim
loss_func = DiceLoss(average=False)
print(loss_func)

optim = Adam(net.parameters(), lr=args.learning_rate)
if os.path.isfile(args.checkpoint):
    print('load optimizer')
    optim.load_state_dict(checkpoint['optim'])
print(optim)

tl = DataLoader(td, batch_size=args.training_batch_size, shuffle=True,
                num_workers=args.num_workers)
vl = DataLoader(vd, batch_size=args.validation_batch_size, shuffle=False,
                num_workers=args.num_workers)

trainer = BasicTrainer(net, loss_func, optim, tl)
tlogger = Logger(args.output_prefix + 'training.csv')
tprinter = Printer('training')
ms = ModelSaver(args.output_prefix)
tp = SegPredSaver(args.output_prefix + 'tra')
trainer.register_observer(tlogger)
trainer.register_observer(tprinter)
trainer.register_observer(ms)
trainer.register_observer(tp)

validator = BasicValidator(vl)
vlogger = Logger(args.output_prefix + 'validation.csv')
vprinter = Printer('validati')
vp = SegPredSaver(args.output_prefix + 'val')
validator.register_observer(vlogger)
validator.register_observer(vprinter)
validator.register_observer(vp)
trainer.register_observer(validator)

trainer.train()
