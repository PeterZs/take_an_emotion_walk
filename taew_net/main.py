import argparse
import os
import numpy as np

from utils import loader, processor

import torch
from torchlight.torchlight import ngpu

import warnings
warnings.filterwarnings("ignore")

base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path, '../datasets')
dataset = 'edin'
coords = 3
if dataset == 'ewalk':
    num_joints = 16
    joint_parents = np.array([-1, 0, 1, 2, 2, 4, 5, 2, 7, 8, 0, 10, 11, 0, 13, 14])
elif dataset == 'edin':
    num_joints = 21
    joint_parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 11, 13, 14, 15, 11, 17, 18, 19])
num_labels = [4, 3, 3]

deep_dim = 14
upsample = 1
model_data_path = os.path.join(base_path, '../weights')


parser = argparse.ArgumentParser(description='Gait Pred')
parser.add_argument('--frame-drop', type=int, default=5, metavar='FD',
                    help='frame downsample rate (default: 1)')
parser.add_argument('--train', type=bool, default=False, metavar='T',
                    help='train the model (default: True)')
parser.add_argument('--load_last_best', type=bool, default=False, metavar='LB',
                    help='load the most recent best model (default: True)')
parser.add_argument('--batch-size', type=int, default=16, metavar='B',
                    help='input batch size for training (default: 32)')
parser.add_argument('--num-worker', type=int, default=4, metavar='W',
                    help='input batch size for training (default: 4)')
parser.add_argument('--start_epoch', type=int, default=0, metavar='SE',
                    help='starting epoch of training (default: 0)')
parser.add_argument('--num_epoch', type=int, default=5000, metavar='NE',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                    help='optimizer (default: Adam)')
parser.add_argument('--base-lr', type=float, default=1e-3, metavar='LR',
                    help='base learning rate (default: 0.1)')
parser.add_argument('--base-tr', type=float, default=1., metavar='TR',
                    help='base teacher rate (default: 1.0)')
parser.add_argument('--step', type=list, default=0.05 * np.arange(20), metavar='[S]',
                    help='fraction of steps when learning rate will be decreased (default: [0.5, 0.75, 0.875])')
parser.add_argument('--lr-decay', type=float, default=0.999, metavar='LRD',
                    help='learning rate decay (default: 0.999)')
parser.add_argument('--tr-decay', type=float, default=0.995, metavar='TRD',
                    help='teacher rate decay (default: 0.995)')
parser.add_argument('--grad_clip', type=float, default=0.2, metavar='GC',
                    help='gradient clip threshold (default: 0.1)')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='D',
                    help='Weight decay (default: 5e-4)')
parser.add_argument('--residual', type=bool, default=False, metavar='R',
                    help='use residual layers (default: True)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='EI',
                    help='interval after which model is evaluated (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='interval after which log is printed (default: 100)')
parser.add_argument('--save-interval', type=int, default=10, metavar='SI',
                    help='interval after which model is saved (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pavi-log', action='store_true', default=False,
                    help='pavi log')
parser.add_argument('--print-log', action='store_true', default=True,
                    help='print log')
parser.add_argument('--save-log', action='store_true', default=True,
                    help='save log')
parser.add_argument('--work-dir', type=str, default=model_data_path, metavar='WD',
                    help='path to save model')
# TO ADD: save_result

args = parser.parse_args()
device = 'cuda:0'

if dataset == 'ewalk':
    [data_train, data_test, poses_train, poses_test, rotations_train, rotations_test,
     translations_train, translations_test, affective_features_train, affective_features_test,
     num_frames_train, num_frames_test, labels_train, labels_test], data_max, data_min =\
        loader.load_ewalk_data(data_path, coords, num_joints, upsample=upsample)
elif dataset == 'edin':
    [data_train, data_test, poses_train, poses_test, rotations_train, rotations_test,
     translations_train, translations_test, affective_features_train, affective_features_test,
     num_frames_train, num_frames_test, labels_train, labels_test], label_weights =\
        loader.load_edin_data(data_path, coords, num_joints, num_labels, frame_drop=args.frame_drop)
diffs_dim = int(rotations_train.shape[-1] / num_joints)
affs_dim = affective_features_train.shape[-1] + deep_dim
affective_features = np.concatenate((affective_features_train, affective_features_test), axis=0)
affective_features, affs_max, affs_min = loader.scale_data(affective_features)
affective_features_train, _, _ = loader.scale_data(affective_features_train, affs_max, affs_min)
affective_features_test, _, _ = loader.scale_data(affective_features_test, affs_max, affs_min)
num_frames_max = rotations_train.shape[1]
num_frames_out = num_frames_max - 1
num_frames_train_norm = num_frames_train / num_frames_max
num_frames_test_norm = num_frames_test / num_frames_max
data_loader = list()
data_loader.append(torch.utils.data.DataLoader(
    dataset=loader.TrainTestLoader(data_train, poses_train, rotations_train, translations_train,
                                   affective_features_train, num_frames_train_norm, labels_train,
                                   coords, num_labels),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_worker * ngpu(device),
    drop_last=True))
data_loader.append(torch.utils.data.DataLoader(
    dataset=loader.TrainTestLoader(data_test, poses_test, rotations_test, translations_test,
                                   affective_features_test, num_frames_test_norm, labels_test,
                                   coords, num_labels),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_worker * ngpu(device),
    drop_last=True))
data_loader = dict(train=data_loader[0], test=data_loader[1])
pr = processor.Processor(args, dataset, data_loader, num_frames_max, num_joints, coords,
                         diffs_dim, affs_dim, num_frames_out, joint_parents, num_labels, affs_max, affs_min,
                         label_weights=label_weights, generate_while_train=False,
                         save_path=base_path, device=device)
if args.train:
    pr.train()
pr.evaluate_model(load_saved_model=True)
