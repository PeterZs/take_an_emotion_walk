import torch.nn as nn

from utils.Quaternions_torch import qeuler
from utils.common import *


def quat_angle_loss(quats_pred, quats_target, V, D):
    quats_in_to_compare = quats_target[:, 1:min(quats_target.shape[1], quats_pred.shape[1] + 1), :]
    quats_in_to_compare = quats_in_to_compare.reshape(-1, quats_in_to_compare.shape[1], V, D)
    quats_out_to_compare = quats_pred[:, :min(quats_target.shape[1] - 1, quats_pred.shape[1]), :]
    quats_out_to_compare = quats_out_to_compare.reshape(-1, quats_out_to_compare.shape[1], V, D)
    euler_in = qeuler(quats_in_to_compare.contiguous(), order='zyx', epsilon=1e-6)
    euler_out = qeuler(quats_out_to_compare.contiguous(), order='zyx', epsilon=1e-6)
    # L1 loss on angle distance with 2pi wrap-around
    angle_distance = torch.remainder(euler_out - euler_in + np.pi, 2 * np.pi) - np.pi
    return torch.mean(torch.abs(angle_distance))


def classifier_loss(labels_pred, labels_target, num_classes, lambda_cls=1., label_weights=None):
    valid_idx = (torch.sum(labels_target, dim=1) > 0).nonzero().squeeze()
    labels_pred_valid = labels_pred[valid_idx, :, :]
    loss = torch.from_numpy(np.array(np.inf)).cuda().float()
    for t in range(labels_pred_valid.shape[1]):
        loss_curr = nn.functional.multilabel_soft_margin_loss(
            labels_pred_valid[:, t, :], labels_target[valid_idx, :],
            weight=label_weights)
        if loss_curr < loss:
            loss = loss_curr
    return lambda_cls * loss
