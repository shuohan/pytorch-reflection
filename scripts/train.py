#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description='Train hierarchical network')

help = ('Configuration .json file. If it is specified and exists, the '
        'command line arguments are ignored.')
parser.add_argument('-j', '--json-config', help=help, default='')

help = ('The data directory. *image.nii are the training images, *label.nii '
        'are the training truth label images, and *mask.nii are ROI masks')
parser.add_argument('-i', '--input-dir', nargs='+', help=help)
parser.add_argument('-o', '--output-prefix', default='',
                    help='Outut model prefix; add slash (/) for folder')
parser.add_argument('-v', '--validation-indices', nargs='+', default=[],
                    help=('The indicies of validation for each dataset, '
                          'comma separate'))
parser.add_argument('-a', '--augmentation', nargs='+', default=list(),
                    choices={'rotate', 'deform', 'scale', 'flip'},
                    help='Data augmentation methods; orders are preserved')
parser.add_argument('-b', '--batch-size', type=int, default=1,
                    help='The number of images per batch')
parser.add_argument('-c', '--cropping-shape', nargs=3, type=int,
                    default=None, help='The shape of the cropped out region')
parser.add_argument('-s', '--input-shape', nargs=3, type=int,
                    default=(192, 256, 192),
                    help='The shape of the input to crop to.')
parser.add_argument('-e', '--num-epochs', type=int, default=200,
                    help='The number of epochs')
parser.add_argument('-p', '--saving-period', type=int, default=10,
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
parser.add_argument('-td', '--training-dir', nargs='+', default=[],
                    help='Training data directories')
parser.add_argument('-vd', '--validation-dir', nargs='+', default=[],
                    help='Validation data directories')
parser.add_argument('-t', '--network-type', choices={'normal', 'lr', 'lr-seg'},
                    default='lr', help='The type of the network.')
parser.add_argument('-is', '--image-suffix', default='image',
                    help='The suffix of the image file.')
parser.add_argument('-ls', '--label-suffix', default='label',
                    help='The suffix of the label image file.')
parser.add_argument('-ms', '--mask-suffix', default='mask',
                    help='The suffix of the mask image file.')
parser.add_argument('-vp', '--validation-period', type=int, default=1,
                    help='The epoch period of validation')
parser.add_argument('-pp', '--pred-saving-period', type=int, default=1,
                    help='Save the prediction every this number of epochs')
parser.add_argument('-ss', '--separate-samples', default=False,
                    action='store_true', help='Print loss separately')

args = parser.parse_args()


import os
import json
import torch
from glob import glob
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import DatasetFactory
from dataset import Config as DatasetConfig
from dataset.pipelines import RandomPipeline
from dataset.workers import WorkerCreator

from pytorch_engine import Config as EngineConfig
from pytorch_engine.training import BasicLogger, Printer, ModelSaver
from pytorch_engine.training import SimpleTrainer, SimpleValidator
from pytorch_engine.training import PredictionSaver
from pytorch_engine.funcs import count_trainable_paras
from pytorch_engine.loss import create_loss

from pytorch_reflection.worker import LabelReorderer


# set configurations
# checkpoint > command arg "json-config" > other args
engine_config = EngineConfig()
dataset_config = DatasetConfig()
if os.path.isfile(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    config = checkpoint['script_config']
    config['checkpoint'] = args.checkpoint
    config['num_epochs'] = args.num_epochs
    config['batch_size'] = args.batch_size
    config['num_workers'] = args.num_workers
    config['saving_period'] = args.saving_period
    # config['input_dir'] = args.input_dir
    # config['output_prefix'] = args.output_prefix
    # config['augmentation'] = args.augmentation
    # config['augmentation_prob'] = args.augmentation_prob
    # config['validation_indices'] = args.validation_indices
    engine_config.update(checkpoint['engine_config'])
    dataset_config.update(checkpoint['dataset_config'])
elif os.path.isfile(args.json_config):
    with open(args.json_config) as json_file:
        config = json.load(json_file)
else:
    config = dict()
for key, value in config.items():
    setattr(args, key, value)
config = args.__dict__

engine_config.decimals = 8
engine_config.eval_separate = args.separate_samples
dataset_config.verbose = args.verbose
dataset_config.aug_prob = args.augmentation_prob
dataset_config.crop_shape = args.cropping_shape
dataset_config.image_shape = args.input_shape
dataset_config.image_suffixes = [args.image_suffix]
dataset_config.label_suffixes = [args.label_suffix]
dataset_config.mask_suffixes = [args.mask_suffix]

# worker_name = 'reorder_label'
# dataset_config.worker_types['addon'].append(worker_name)
# WorkerCreator().register(worker_name, LabelReorderer)
# print(WorkerCreator())

# load datasets
ds_factory = DatasetFactory()
ds_factory.add_image_type('image', 'label')
if args.cropping_shape is not None:
    ds_factory.add_image_type('mask')

if args.input_dir:
    len_diff = len(args.input_dir) - len(args.validation_indices)
    args.validation_indices += [None] * len_diff
    val_inds = list()
    for i, (indir, val_ind) in enumerate(zip(args.input_dir,
                                             args.validation_indices)):
        if val_ind is not None:
            val_ind = [int(num) for num in val_ind.split(',')]
        val_inds.append(val_ind)
        ds_factory.add_dataset(dataset_id = str(i),
                               dirname=indir, val_ind=val_ind)
elif args.training_dir:
    len_diff = len(args.training_dir) - len(args.validation_dir)
    args.validation_dir += [None] * len_diff
    for i, (tra_dir, val_dir) in enumerate(zip(args.training_dir,
                                               args.validation_dir)):
        ds_factory.add_dataset(dataset_id = str(i),
                               t_dirname=tra_dir, v_dirname=val_dir)
else:
    raise RuntimeError('input-dir or train-dir should not be empty')

ds_factory.add_training_operation('resize')
ds_factory.add_validation_operation('resize')
ds_factory.add_training_operation(*args.augmentation)
if args.cropping_shape is not None:
    ds_factory.add_training_operation('crop')
    ds_factory.add_validation_operation('crop')

# if args.network_type == 'lr-seg':
#     ds_factory.add_training_operation('reorder_label')
#     ds_factory.add_validation_operation('reorder_label')
# else:
ds_factory.add_training_operation('norm_label')
ds_factory.add_validation_operation('norm_label')

t_dataset, v_dataset = ds_factory.create()

if 'flip' in args.augmentation:
    augmentation = args.augmentation.copy()
    augmentation.remove('flip')
    pipeline = RandomPipeline()
    pipeline.register('resize')
    pipeline.register(*augmentation)
    if args.cropping_shape is not None:
        pipeline.register('crop')
    # if args.network_type == 'lr-seg':
    #     pipeline.register('reorder_label')
    # else:
    pipeline.register('norm_label')
    t_dataset.add_pipeline(pipeline)

pipeline = RandomPipeline()
pipeline.register('resize')
pipeline.register('flip')
if args.cropping_shape is not None:
    pipeline.register('crop')
# if args.network_type == 'lr-seg':
#     pipeline.register('reorder_label')
# else:
pipeline.register('norm_label')
v_dataset.add_pipeline(pipeline)

# print datasets
print('-' * 80)
print('training dataset')
print('# training data', len(t_dataset))
print(t_dataset)
print('-' * 80)
print('validation dataset')
print('# validation data', len(v_dataset))
print(v_dataset)

t_loader = DataLoader(t_dataset, batch_size=args.batch_size, shuffle=True,
                      num_workers=args.num_workers)

if args.network_type == 'lr':
    from pytorch_reflection.unet import LRUNet as Net
elif args.network_type == 'normal':
    from pytorch_reflection.unet import UNet as Net
else:
    from pytorch_reflection.unet import LRSegUNet as Net


out_classes = len(t_dataset.labels)
if out_classes == 2:
    out_classes -= 1
if args.network_type == 'lr-seg':
    label_image = t_dataset.images[0][1].normalize()
    paired_labels = label_image.pairs
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
loss_func = create_loss()

# if args.network_type == 'lr-seg':
#     from pytorch_reflection.loss import DiceLoss
# else:
#     from pytorch_engine.loss import DiceLoss
# loss_func = DiceLoss()
# print(loss_func)

lr = args.learning_rate * args.batch_size
optimizer = Adam(net.parameters(), lr=lr)
print(optimizer)

num_batches = len(t_dataset) // args.batch_size \
            + ((len(t_dataset) % args.batch_size) > 0)
# trainer
trainer = SimpleTrainer(net, loss_func, optimizer, t_loader,
                        num_epochs=args.num_epochs)
printer = Printer('training')
trainer.register_observer(printer)
saver = ModelSaver(args.saving_period, args.output_prefix,
                   dataset_config=dataset_config, script_config=config)
trainer.register_observer(saver)
t_logger = BasicLogger(args.output_prefix + 'training.csv')
trainer.register_observer(t_logger)
tsaver = PredictionSaver(args.pred_saving_period, args.output_prefix+'tra_')
trainer.register_observer(tsaver)

# validator
if len(v_dataset) > 0:
    v_loader = DataLoader(v_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)
    validator = SimpleValidator(v_loader, saving_period=args.validation_period)

    v_printer = Printer('validation')
    v_logger = BasicLogger(args.output_prefix + 'validation.csv')

    vsaver = PredictionSaver(args.pred_saving_period, args.output_prefix+'val_')
    validator.register_observer(v_logger)
    validator.register_observer(v_printer)
    validator.register_observer(vsaver)

    trainer.register_observer(validator)

# print configurations
print('-' * 80)
print('Engine configurations')
print(engine_config)
print('-' * 80)
print('Dataset configurations')
print(dataset_config)
print('-' * 80)
print('Script configurations')
keylen = max([len(key)+1 for key in config.keys()])
for key, value in config.items():
    print('    %s %s' % ((key+':').ljust(keylen), value))

# train model
trainer.train()
