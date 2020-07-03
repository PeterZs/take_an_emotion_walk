import os
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch

def load_data(data_path, frame_drop=5):
    # load
    npzfile = np.load(data_path)
    joints = npzfile['joints']
    bones = npzfile['bones']

    # frame drop
    joints = joints[:, ::frame_drop, :]
    bones = bones[:, ::frame_drop, :]

    # reshape
    joints = joints.reshape(joints.shape[0], joints.shape[1], 21, 3)

    labels = npzfile['labels']
    labels = labels / np.linalg.norm(labels, ord=1, axis=1)[:, None]

    return train_test_split(joints, bones, labels, test_size=0.1, shuffle=True, random_state=420)


def create_loader(args):
    if args.stgcn or args.dgnn:
        joints_train, joints_test, bones_train, bones_test, labels_train, labels_test = load_data(
            "datasets/elmd_dag.npz")
        joints_val, bones_val, Y_val = torch.from_numpy(joints_test).cuda(), torch.from_numpy(
            bones_test).cuda(), torch.from_numpy(labels_test).cuda()
        # only need first 4 labels
        Y_val = Y_val[:, :4]

        # train_set = TensorDataset(joints_tr, bones_tr, Y_train)
        val_set = TensorDataset(joints_val, bones_val, Y_val)

        # train_loader = DataLoader(train_set, batch_size=, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=161, shuffle=True)
        print("Loaded: %d training, %d val" % (joints_train.shape[0], joints_val.shape[0]))

        return val_loader

    else:
        print("Model not implemented or too many options, exiting")
        exit()
