import argparse
import os
import numpy as np
import sklearn.metrics as skm

from utils import loader, processor
from utils.common import to_multi_hot

from torch.utils.data import TensorDataset, DataLoader

import torch
import torchlight


base_path = os.path.dirname(os.path.realpath(__file__))
coords = 3
joints = 21
cycles = 1
model_path = os.path.join(base_path, '../weights')


parser = argparse.ArgumentParser(description='Gait Pred')
parser.add_argument('--train', type=bool, default=False, metavar='T',
                    help='train the model (default: True)')
parser.add_argument('--smap', type=bool, default=False, metavar='S',
                    help='train the model (default: True)')
parser.add_argument('--save-features', type=bool, default=False, metavar='SF',
                    help='save penultimate layer features (default: True)')
parser.add_argument('--batch-size', type=int, default=8, metavar='B',
                    help='input batch size for training (default: 8)')
parser.add_argument('--num-worker', type=int, default=4, metavar='W',
                    help='input batch size for training (default: 4)')
parser.add_argument('--start_epoch', type=int, default=0, metavar='SE',
                    help='starting epoch of training (default: 0)')
parser.add_argument('--num_epoch', type=int, default=500, metavar='NE',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                    help='optimizer (default: Adam)')
parser.add_argument('--base-lr', type=float, default=0.1, metavar='L',
                    help='base learning rate (default: 0.1)')
parser.add_argument('--step', type=list, default=[0.5, 0.75, 0.875], metavar='[S]',
                    help='fraction of steps when learning rate will be decreased (default: [0.5, 0.75, 0.875])')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='D',
                    help='Weight decay (default: 5e-4)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='EI',
                    help='interval after which model is evaluated (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='interval after which log is printed (default: 100)')
parser.add_argument('--topk', type=list, default=[1], metavar='[K]',
                    help='top K accuracy to show (default: [1])')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pavi-log', action='store_true', default=False,
                    help='pavi log')
parser.add_argument('--print-log', action='store_true', default=True,
                    help='print log')
parser.add_argument('--save-log', action='store_true', default=True,
                    help='save log')
parser.add_argument('--work-dir', type=str, default=model_path, metavar='WD',
                    help='path to save')
parser.add_argument('--frame_drop', type=int, default=5, metavar='FD',
                    help='frame downsample rate (default: 1)')  # CAHNGE THIS, TRY WITH 3 and 5
# TO ADD: save_result

args = parser.parse_args()
device = 'cuda:0'

num_joints = 21
num_labels = [4, 3, 3]
num_classes = 4

data, labels, [data_train, data_test, labels_train, labels_test] =\
        loader.load_edin_data(
            'datasets/data_edin_locomotion_pose_diff_aff_drop_{}.npz'.format(args.frame_drop),
            'datasets/labels_edin_locomotion', num_labels)
graph_dict = {'strategy': 'spatial'}
max_time_steps = data.shape[1]

if args.train:
    X_train, X_val = torch.from_numpy(data_train).cuda(), torch.from_numpy(data_test).cuda()
    Y_train, Y_val = torch.from_numpy(labels_train).cuda(), torch.from_numpy(labels_test).cuda()

    train_set = TensorDataset(X_train, Y_train)
    val_set = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_set, batch_size=128)
    val_loader = DataLoader(val_set, batch_size=128)

    data_loader_train_test = dict(train=train_loader, test=val_loader)
    print('Train set size: {:d}'.format(len(data_train)))
    print('Test set size: {:d}'.format(len(data_test)))
    print('Number of classes: {:d}'.format(num_classes))
    pr = processor.Processor(args, data_loader_train_test, num_joints, coords, max_time_steps, num_classes, graph_dict,
                             device=device, verbose=True)
    pr.train()
else:
    pr = processor.Processor(args, None, num_joints, coords, max_time_steps, num_classes, graph_dict,
                             device=device, verbose=False)
    preds = pr.generate_predictions(data_test, num_labels[0], joints, coords)
    labels_pred = to_multi_hot(preds)
    labels_true = to_multi_hot(labels_test)
    aps = skm.average_precision_score(labels_true, labels_pred, average=None)
    mean_ap = skm.average_precision_score(labels_true, labels_pred, average='micro')
    print('aps: {}'.format(aps))
    print('mean ap: {}'.format(mean_ap))
    # for idx in range(labels_pred.shape[0]):
    #     print('{:d}.\t{:s}'.format(idx, emotions[int(labels_pred[idx])]))
    # if args.smap:
    #     pr.smap()
