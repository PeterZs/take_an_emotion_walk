import h5py
import math
import os
import matplotlib.pylab as plt
import numpy as np
import sklearn.metrics as skm
import torch
import torch.optim as optim
import torch.nn as nn
from net import hap
import random

from torchlight.torchlight.io import IO
from utils.common import reconstruct_gait, to_multi_hot
from utils.visualizations import display_animations
from utils import loader
from utils import losses
from utils.common import *

torch.manual_seed(1234)

rec_loss = losses.quat_angle_loss


def h5_to_csv(h5file, csv_save_path):
    f = h5py.File(h5file, 'r')
    for idx in range(len(f.keys())):
        a_group_key = list(f.keys())[idx]
        data = np.array(f[a_group_key])  # Get the data
        np.savetxt(os.path.join(csv_save_path, a_group_key + '.csv'), data, delimiter=',')  # Save as csv


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.05)
        if m.bias is not None:
            m.bias.data.normal_(0.0, 0.05)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm1d') != -1:
        m.weight.data.normal_(0.2, 0.02)
        m.bias.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        if m.out_features < 32:
            m.weight.data.normal_(0.5, 0.1)
            if m.bias is not None:
                m.bias.data.normal_(0.5, 0.1)
        else:
            m.weight.data.normal_(0.0, 0.05)
            if m.bias is not None:
                m.bias.data.normal_(0.0, 0.05)
    elif classname.find('ModuleList') != -1:
        for l in m:
            subclassname = l.__class__.__name__
            if subclassname.find('Conv1d') != -1:
                l.weight.data.normal_(0.0, 0.05)
                if l.bias is not None:
                    l.bias.data.normal_(0.0, 0.05)
            elif subclassname.find('Conv2d') != -1:
                l.weight.data.normal_(0.0, 0.02)
                if l.bias is not None:
                    l.bias.data.fill_(0)
            elif subclassname.find('Linear') != -1:
                if l.out_features < 32:
                    l.weight.data.normal_(0.5, 0.1)
                    if l.bias is not None:
                        l.bias.data.normal_(0.5, 0.1)
                else:
                    l.weight.data.normal_(0.0, 0.07)
                    if l.bias is not None:
                        l.bias.data.normal_(0.0, 0.07)


def calculate_metrics(y_true, y_pred, thres=0.25, eval_time=False):
    valid_idx = np.squeeze(np.argwhere(np.sum(y_true, axis=1)))
    y_true = y_true[valid_idx, :]
    y_pred = y_pred[valid_idx, :, :]
    ap = np.zeros((y_pred.shape[1], y_pred.shape[2]))
    mean_ap = np.zeros(y_pred.shape[1])
    acc = np.zeros(y_pred.shape[1])
    f1_scores = np.zeros((y_pred.shape[1], y_pred.shape[2]))
    for t in range(y_pred.shape[1]):
        ap[t, :] = skm.average_precision_score(y_true, y_pred[:, t, :], average=None)
        mean_ap[t] = skm.average_precision_score(y_true.reshape(-1), y_pred[:, t, :].reshape(-1), average='micro')
        if np.isnan(mean_ap[t]):
            mean_ap[t] = skm.average_precision_score(y_true, y_pred[:, t, :], average='micro')
        y_pred_multi_hot = to_multi_hot(y_pred[:, t, :], threshold=thres)
        _, _, f1_scores[t, :], _ = skm.precision_recall_fscore_support(y_true, y_pred_multi_hot, average=None)
        acc[t] = 1. - skm.hamming_loss(y_true, y_pred_multi_hot)
    best_ap = np.max(ap, axis=0)
    total_nans = np.sum(np.isnan(best_ap))
    best_mean_ap = np.sum(np.nan_to_num(best_ap)) / (len(best_ap) - total_nans)
    return np.nan_to_num(best_ap), best_mean_ap, np.mean(np.max(f1_scores, axis=0)), np.max(acc)


def semisup_loss(l, l_pred, x_in, x_out, x_out_pre_norm, epoch, affs, affs_pred, V, D, num_classes,
                 lambda_ang=1., lambda_pen=0.5, lambda_aff=0.01, lambda_cls=1., label_weights=None, eval_time=False):
    rec = lambda_ang * rec_loss(x_out.float(), x_in.permute(0, 2, 1).cuda().float(), V, D)
    pen = lambda_pen * torch.mean((torch.sum(x_out_pre_norm**2, dim=3) - 1)**2)
    affs_dim = affs.shape[-1]
    aff_loss = lambda_aff * torch.mean((torch.mean((affs[:, 1:, :] -
                                                   nn.functional.sigmoid(affs_pred[:, :, :affs_dim])) ** 2,
                                                  dim=1)) ** 2)
    classifier_loss = losses.classifier_loss(l_pred[0], l[0].cuda().float(), num_classes,
                                             lambda_cls=lambda_cls, label_weights=label_weights)
    return rec + pen + aff_loss + classifier_loss


def find_all_substr(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_best_epoch_loss_and_acc(path_to_model_files):
    all_models = os.listdir(path_to_model_files)
    if len(all_models) < 2:
        return 0, np.inf
    loss_list = -1. * np.ones(len(all_models))
    acc_list = -1. * np.ones(len(all_models))
    for i, model in enumerate(all_models):
        loss_acc_val = str.split(model, '_')
        if len(loss_acc_val) > 1:
            loss_list[i] = float(loss_acc_val[3])
            acc_list[i] = float(loss_acc_val[5])
    if len(loss_list) < 3:
        best_model = all_models[np.argwhere(loss_list == min([n for n in loss_list if n > 0]))[0, 0]]
    else:
        loss_idx = np.argpartition(loss_list, 2)
        best_model = all_models[loss_idx[1]]
    all_underscores = list(find_all_substr(best_model, '_'))
    # return model name, best loss, best acc
    return best_model, int(best_model[all_underscores[0] + 1:all_underscores[1]]),\
           float(best_model[all_underscores[2] + 1:all_underscores[3]]),\
           float(best_model[all_underscores[4] + 1:all_underscores[5]])


class Processor(object):
    """
        Processor for gait generation
    """

    def __init__(self, args, dataset, data_loader, T, V, C, D, A, T_out, joint_parents, num_classes,
                 affs_max, affs_min,
                 min_train_epochs=-1, label_weights=None, generate_while_train=False, poses_mean=None,
                 poses_std=None, save_path=None, device='cuda:0'):

        self.args = args
        self.device = device
        self.dataset = dataset
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.affs_max = affs_max
        self.affs_min = affs_min
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.io = IO(
            self.args.work_dir,
            save_log=self.args.save_log,
            print_log=self.args.print_log)

        # model
        self.T = T
        self.V = V
        self.C = C
        self.D = D
        self.A = A
        self.T_out = T_out
        self.P = int(0.9 * T)
        self.joint_parents = joint_parents
        self.model = hap.HAPPY(self.dataset, T, V, C, A, T_out, num_classes, residual=self.args.residual)
        self.model.cuda(device)
        self.model.apply(weights_init)
        self.model_GRU_h_enc = None
        self.model_GRU_h_dec1 = None
        self.model_GRU_h_dec = None
        self.label_weights = torch.from_numpy(label_weights).cuda().float()
        self.loss = semisup_loss
        self.best_loss = math.inf
        self.best_mean_ap = 0.
        self.loss_updated = False
        self.mean_ap_updated = False
        self.step_epochs = [math.ceil(float(self.args.num_epoch * x)) for x in self.args.step]
        self.best_loss_epoch = None
        self.best_acc_epoch = None
        self.min_train_epochs = min_train_epochs
        self.beta = 0.1

        # generate
        self.generate_while_train = generate_while_train
        self.poses_mean = poses_mean
        self.poses_std = poses_std
        self.save_path = save_path
        self.dataset = dataset

        # optimizer
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr)
                # weight_decay=self.args.weight_decay)
        else:
            raise ValueError()
        self.lr = self.args.base_lr
        self.tr = self.args.base_tr

    def process_data(self, data, poses, diffs, affs):
        data = data.float().to(self.device)
        poses = poses.float().to(self.device)
        diffs = diffs.float().to(self.device)
        affs = affs.float().to(self.device)
        return data, poses, diffs, affs

    def load_best_model(self, ):
            loaded_vars = torch.load(os.path.join(self.args.work_dir, 'taew_weights.pth.tar'))
            self.model.load_state_dict(loaded_vars['model_dict'])
            self.model_GRU_h_enc = loaded_vars['h_enc']
            self.model_GRU_h_dec1 = loaded_vars['h_dec1']
            self.model_GRU_h_dec = loaded_vars['h_dec']

    def adjust_lr(self):
        self.lr = self.lr * self.args.lr_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def adjust_tr(self):
        self.tr = self.tr * self.args.tr_decay

    def show_epoch_info(self, show_best=True):

        print_epochs = [self.best_loss_epoch if self.best_loss_epoch is not None else 0,
                        self.best_acc_epoch if self.best_acc_epoch is not None else 0,
                        self.best_acc_epoch if self.best_acc_epoch is not None else 0]
        best_metrics = [self.best_loss, 0, self.best_mean_ap]
        i = 0
        for k, v in self.epoch_info.items():
            if show_best:
                self.io.print_log('\t{}: {}. Best so far: {} (epoch: {:d}).'.
                                  format(k, v, best_metrics[i], print_epochs[i]))
            else:
                self.io.print_log('\t{}: {}.'.format(k, v))
            i += 1
        if self.args.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):

        if self.meta_info['iter'] % self.args.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.args.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def per_train(self):

        self.model.train()
        train_loader = self.data_loader['train']
        loss_value = []
        ap_values = []
        mean_ap_value = []

        for data, poses, rots, affs, num_frames, labels in train_loader:
            # get data
            num_frames_actual, sort_idx = torch.sort((num_frames * self.T).type(torch.IntTensor).cuda(),
                                                     descending=True)
            seq_lens = num_frames_actual - 1
            data = data[sort_idx, :, :]
            poses = poses[sort_idx, :]
            rots = rots[sort_idx, :, :]
            affs = affs[sort_idx, :]
            data, poses, rots, affs = self.process_data(data, poses, rots, affs)

            # forward
            labels_pred, diffs_recons, diffs_recons_pre_norm, affs_pred,\
            self.model_GRU_h_enc, self.model_GRU_h_dec1, self.model_GRU_h_dec =\
                self.model(poses, rots[:, :, 1:], affs, teacher_steps=int(self.T * self.tr))
            loss = self.loss(labels, labels_pred, rots, diffs_recons, diffs_recons_pre_norm, self.meta_info['epoch'],
                             affs, affs_pred, self.V, self.D, self.num_classes[0], label_weights=self.label_weights)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['aps'], self.iter_info['mean_ap'], self.iter_info['f1'], _ =\
                calculate_metrics(labels[0].detach().cpu().numpy(),
                                  labels_pred[0].detach().cpu().numpy())
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.iter_info['tr'] = '{:.6f}'.format(self.tr)
            loss_value.append(self.iter_info['loss'])
            ap_values.append(self.iter_info['aps'])
            mean_ap_value.append(self.iter_info['mean_ap'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['mean_aps'] = np.mean(ap_values, axis=0)
        self.epoch_info['mean_mean_ap'] = np.mean(mean_ap_value)
        self.show_epoch_info()
        self.io.print_timer()
        self.adjust_lr()
        self.adjust_tr()

    def per_test(self, epoch=None, evaluation=True):

        self.model.eval()
        test_loader = self.data_loader['test']
        loss_value = []
        mean_ap_value = []
        ap_values = []
        label_frag = []

        for data, poses, diffs, affs, num_frames, labels in test_loader:
            # get data
            num_frames_actual, sort_idx = torch.sort((num_frames * self.T).type(torch.IntTensor).cuda(),
                                                     descending=True)
            seq_lens = num_frames_actual - 1
            data = data[sort_idx, :]
            poses = poses[sort_idx, :]
            diffs = diffs[sort_idx, :, :]
            affs = affs[sort_idx, :]
            data, poses, diffs, affs = self.process_data(data, poses, diffs, affs)

            # inference
            with torch.no_grad():
                labels_pred, diffs_recons, diffs_recons_pre_norm, affs_pred, _, _, _ = \
                    self.model(poses, diffs[:, :, 1:], affs,
                               teacher_steps=int(self.T * self.tr))

            # get loss
            if evaluation:
                loss = self.loss(labels, labels_pred, diffs, diffs_recons, diffs_recons_pre_norm, self.meta_info['epoch'],
                                 affs, affs_pred, self.V, self.D, self.num_classes[0],
                                 label_weights=self.label_weights, eval_time=True)
                loss_value.append(loss.item())
                ap, mean_ap, _, _ = calculate_metrics(labels[0].detach().cpu().numpy(),
                                                       labels_pred[0].detach().cpu().numpy(), eval_time=True)
                ap_values.append(ap)
                mean_ap_value.append(mean_ap)

                label_frag.append(labels[0].data.cpu().numpy())

        if evaluation:
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.epoch_info['mean_aps'] = np.mean(ap_values, axis=0)
            self.epoch_info['mean_mean_ap'] = np.mean(mean_ap_value)
            if self.epoch_info['mean_loss'] < self.best_loss and epoch > self.min_train_epochs:
                self.best_loss = self.epoch_info['mean_loss']
                self.best_loss_epoch = self.meta_info['epoch']
                self.loss_updated = True
            else:
                self.loss_updated = False
            if self.epoch_info['mean_mean_ap'] > self.best_mean_ap and epoch > self.min_train_epochs:
                self.best_mean_ap = self.epoch_info['mean_mean_ap']
                self.best_acc_epoch = self.meta_info['epoch']
                self.mean_ap_updated = True
            else:
                self.mean_ap_updated = False
            self.show_epoch_info()

    def train(self):

        if self.args.load_last_best:
            self.load_best_model()
            self.args.start_epoch = self.best_loss_epoch
        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            self.meta_info['epoch'] = epoch

            # training
            self.io.print_log('Training epoch: {}'.format(epoch))
            self.per_train()
            self.io.print_log('Done.')

            # evaluation
            if (epoch % self.args.eval_interval == 0) or (
                    epoch + 1 == self.args.num_epoch):
                self.io.print_log('Eval epoch: {}'.format(epoch))
                self.per_test(epoch)
                self.io.print_log('Done.')

            # save model and weights
            if self.loss_updated or self.mean_ap_updated:
                torch.save({'model_dict': self.model.state_dict(),
                            'h_enc': self.model_GRU_h_enc,
                            'h_dec1': self.model_GRU_h_dec1,
                            'h_dec': self.model_GRU_h_dec},
                           os.path.join(self.args.work_dir, 'epoch_{}_loss_{:.4f}_acc_{:.2f}_model.pth.tar'.
                                        format(epoch, self.best_loss, self.best_mean_ap * 100.)))

                if self.generate_while_train:
                    self.generate(load_saved_model=False, samples_to_generate=1)

    def test(self):

        # the path of weights must be appointed
        if self.args.weights is None:
            raise ValueError('Please appoint --weights.')
        self.io.print_log('Model:   {}.'.format(self.args.model))
        self.io.print_log('Weights: {}.'.format(self.args.weights))

        # evaluation
        self.io.print_log('Evaluation Start:')
        self.per_test()
        self.io.print_log('Done.\n')

        # save the output of model
        if self.args.save_result:
            result_dict = dict(
                zip(self.data_loader['test'].dataset.sample_name,
                    self.result))
            self.io.save_pkl(result_dict, 'test_result.pkl')

    def evaluate_model(self, load_saved_model=True):
        if load_saved_model:
            self.load_best_model()
        self.model.eval()
        test_loader = self.data_loader['test']
        loss_value = []
        mean_ap_value = []
        ap_values = []
        label_frag = []

        for data, poses, diffs, affs, num_frames, labels in test_loader:
            # get data
            num_frames_actual, sort_idx = torch.sort((num_frames * self.T).type(torch.IntTensor).cuda(),
                                                     descending=True)
            seq_lens = num_frames_actual - 1
            data = data[sort_idx, :]
            poses = poses[sort_idx, :]
            diffs = diffs[sort_idx, :, :]
            affs = affs[sort_idx, :]
            data, poses, diffs, affs = self.process_data(data, poses, diffs, affs)

            # inference
            with torch.no_grad():
                labels_pred, diffs_recons, diffs_recons_pre_norm, affs_pred, _, _, _ = \
                    self.model(poses, diffs[:, :, 1:], affs,
                               teacher_steps=int(self.T * self.tr))

            # get loss
            loss = self.loss(labels, labels_pred, diffs, diffs_recons, diffs_recons_pre_norm,
                             self.meta_info['epoch'],
                             affs, affs_pred, self.V, self.D, self.num_classes[0],
                             label_weights=self.label_weights, eval_time=True)
            loss_value.append(loss.item())
            ap, mean_ap, _, _ = calculate_metrics(labels[0].detach().cpu().numpy(),
                                                  labels_pred[0].detach().cpu().numpy(), eval_time=True)
            ap_values.append(ap)
            mean_ap_value.append(mean_ap)

            label_frag.append(labels[0].data.cpu().numpy())

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['aps'] = np.mean(ap_values, axis=0)
        self.epoch_info['mean_ap'] = np.mean(mean_ap_value)
        self.show_epoch_info(show_best=False)
