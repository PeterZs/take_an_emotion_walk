# sys
import csv
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from scipy import signal
from sklearn.model_selection import train_test_split
from torchlight.torchlight import IO
from utils.Quaternions import Quaternions
from utils import common

# torch
import torch
from torchvision import datasets, transforms


def load_ewalk_data(_path, coords, joints, upsample=1):

    file_feature = os.path.join(_path, 'features' + '_ewalk.h5')
    ff = h5py.File(file_feature, 'r')
    file_label = os.path.join(_path, 'labels' + '_ewalk.h5')
    fl = h5py.File(file_label, 'r')

    data_list = []
    num_samples = len(ff.keys())
    time_steps = 0
    labels = np.empty(num_samples)
    for si in range(num_samples):
        ff_group_key = list(ff.keys())[si]
        data_list.append(list(ff[ff_group_key]))  # Get the data
        time_steps_curr = len(ff[ff_group_key])
        if time_steps_curr > time_steps:
            time_steps = time_steps_curr
        labels[si] = fl[list(fl.keys())[si]][()]

    data = np.zeros((num_samples, time_steps * upsample, joints * coords))
    num_frames = np.empty(num_samples)
    for si in range(num_samples):
        data_list_curr_arr = np.array(data_list[si])
        tsteps_curr = len(data_list[si]) * upsample
        for lidx in range(data_list_curr_arr.shape[1]):
            data[si, :tsteps_curr, lidx] = signal.resample(data_list_curr_arr[:, lidx], tsteps_curr)
            if lidx > 0 and lidx % coords == 0:
                temp = np.copy(data[si, :tsteps_curr, lidx - 1])
                data[si, :tsteps_curr, lidx - 1] = np.copy(- data[si, :tsteps_curr, lidx - 2])
                data[si, :tsteps_curr, lidx - 2] = temp
                rotation = Quaternions.from_angle_axis(np.pi / 2., np.array([1, 0, 0]))
                for t in range(tsteps_curr):
                    data[si, t, lidx - 3:lidx] = rotation * data[si, t, lidx - 3:lidx]
        num_frames[si] = tsteps_curr
    poses, differentials, affective_features = common.get_ewalk_differentials_with_padding(data, num_frames, coords)
    return train_test_split(poses, differentials, affective_features, num_frames, labels, test_size=0.1)


def load_edin_labels(_path, num_labels):
    labels_dir = os.path.join(_path)
    annotators = os.listdir(labels_dir)
    num_annotators = len(annotators)
    labels = np.zeros((num_labels, num_annotators))

    for file in annotators:
        with open(os.path.join(labels_dir, file)) as csv_file:
            read_line = csv.reader(csv_file, delimiter=',')
            row_count = -1
            for row in read_line:
                row_count += 1
                if row_count == 0:
                    continue
                try:
                    data_idx = int(row[0].split('_')[-1])
                except ValueError:
                    data_idx = row_count - 1
                emotion = row[1].split(sep=' ')
                behavior = row[2].split(sep=' ')
                personality = row[3].split(sep=' ')
                if len(emotion) == 1 and emotion[0].lower() == 'neutral':
                    labels[data_idx, 3] += 1.
                elif len(emotion) > 1:
                    counter = 0.
                    if emotion[0].lower() == 'extremely':
                        counter = 1.
                    elif emotion[0].lower() == 'somewhat':
                        counter = 1.
                    if emotion[1].lower() == 'happy':
                        labels[data_idx, 0] += counter
                    elif emotion[1].lower() == 'sad':
                        labels[data_idx, 1] += counter
                    elif emotion[1].lower() == 'angry':
                        labels[data_idx, 2] += counter
                if len(behavior) == 1 and behavior[0].lower() == 'neutral':
                    labels[data_idx, 6] += 1.
                elif len(behavior) > 1:
                    counter = 0.
                    if behavior[0].lower() == 'highly':
                        counter = 2.
                    elif behavior[0].lower() == 'somewhat':
                        counter = 1.
                    if behavior[1].lower() == 'dominant':
                        labels[data_idx, 4] += counter
                    elif behavior[1].lower() == 'submissive':
                        labels[data_idx, 5] += counter
                if len(personality) == 1 and personality[0].lower() == 'neutral':
                    labels[data_idx, 9] += 1.
                elif len(personality) > 1:
                    counter = 0.
                    if personality[0].lower() == 'extremely':
                        counter = 2.
                    elif personality[0].lower() == 'somewhat':
                        counter = 1.
                    if personality[1].lower() == 'friendly':
                        labels[data_idx, 7] += counter
                    elif personality[1].lower() == 'unfriendly':
                        labels[data_idx, 8] += counter
    return labels, num_annotators


def load_edin_data(data_path, label_dir_path, num_labels, frame_drop=1):

    npzfile = np.load(data_path)
    data = npzfile['arr_0']
    data = data.reshape(data.shape[0], data.shape[1], 21, 3)
    data = data[:, ::frame_drop, :]
    labels, num_annotators = load_edin_labels(label_dir_path, data.shape[0])
    labels /= (num_annotators * 2.)
    label_partitions = np.append([0], np.cumsum(num_labels))
    for lpidx in range(len(num_labels)):
        labels[:, label_partitions[lpidx]:label_partitions[lpidx + 1]] =\
            labels[:, label_partitions[lpidx]:label_partitions[lpidx + 1]] /\
            np.linalg.norm(labels[:, label_partitions[lpidx]:label_partitions[lpidx + 1]], ord=1, axis=1)[:, None]
    num_samples = labels.shape[0]
    # balanced_emo_idx = balance_edin_emotion_labels(labels)
    # data = data[balanced_emo_idx, :, :]
    # labels = labels[balanced_emo_idx, :]
    # data = data[:-num_samples, :, :]
    # labels = labels[:-num_samples, :]
    labels = labels[:, :4]
    return data, labels, train_test_split(data, labels, test_size=0.1)


def balance_edin_emotion_labels(labels):
    labels_emo = labels[:, :4]
    labels_oh = np.zeros_like(labels_emo)
    labels_oh[np.arange(len(labels_emo)), labels_emo.argmax(1)] = 1
    c_idx = []
    c_idx.append(np.squeeze(np.argwhere(labels_oh[:, 0] == 1)))
    c_idx.append(np.squeeze(np.argwhere(labels_oh[:, 1] == 1)))
    c_idx.append(np.squeeze(np.argwhere(labels_oh[:, 2] == 1)))
    c_idx.append(np.squeeze(np.argwhere(labels_oh[:, 3] == 1)))
    return np.concatenate((c_idx[0], np.repeat(c_idx[1], 1), np.repeat(c_idx[2], 2), np.repeat(c_idx[3], 4)), axis=0)


def recalculate_edin_affective_features(data, num_frames, coords):
    num_samples = data.shape[0]
    affs_dim = 18
    affective_features = np.zeros((num_samples, affs_dim))
    for sidx in range(num_samples):
        affective_features_curr = np.zeros((np.max(num_frames), affs_dim))
        for tidx in range(int(num_frames[sidx])):
            affective_features_curr[tidx, :] = \
                common.get_edin_affective_features(data[sidx, tidx, :], coords, affs_dim)
        affective_features[sidx, :] = np.mean(affective_features_curr, axis=0)

    return affective_features


def plot_edin_affective_features_class_wise(affective_features, labels):
    labels_oh = np.zeros_like(labels)
    labels_oh[np.arange(len(labels)), labels.argmax(1)] = 1
    c_idx = []
    c_idx.append(np.squeeze(np.argwhere(labels_oh[:, 0] == 1)))
    c_idx.append(np.squeeze(np.argwhere(labels_oh[:, 1] == 1)))
    c_idx.append(np.squeeze(np.argwhere(labels_oh[:, 2] == 1)))
    c_idx.append(np.squeeze(np.argwhere(labels_oh[:, 3] == 1)))
    for aidx in range(affective_features.shape[1]):
        save_path = 'plots/' + str(aidx) + '.png'
        for c in c_idx:
            affs_curr = affective_features[c, aidx]
            plt.hist(affs_curr)
        plt.grid()
        plt.legend(['0', '1', '2', '3'])
        plt.savefig(os.path.join(save_path))
        plt.cla()


def normalize_data(_data, data_mean=None, data_std=None):
    _data = _data.astype('float32')
    if data_mean is None:
        data_mean = np.mean(_data, axis=0)
    if data_std is None:
        data_std = np.std(_data, axis=0)
    return (_data - data_mean) / data_std, data_mean, data_std


def normalize_data_per_frame(_data, _num_frames):
    _data = _data.astype('float32')
    data_collated = _data[0, :int(_num_frames[0]), :]
    for idx in range(1, len(_data)):
        data_collated = np.append(data_collated, _data[idx, :int(_num_frames[idx]), :], axis=0)
    data_mean = np.mean(data_collated)
    data_std = np.std(data_collated)
    data_normalized = np.zeros_like(_data)
    incr_frame_num = 0
    for idx in range(len(_data)):
        data_normalized[idx, :int(_num_frames[idx]), :] =\
            (data_collated[incr_frame_num:incr_frame_num + int(_num_frames[idx]), :] - data_mean) / data_std
        incr_frame_num += int(_num_frames[idx])
    return data_normalized, data_mean, data_std


def scale_data(_data, data_max=None, data_min=None):
    _data = _data.astype('float32')
    if data_max is None:
        data_max = np.max(_data)
    if data_min is None:
        data_min = np.min(_data)
    return (_data - data_min) / (data_max - data_min), data_max, data_min


def scale_per_joint(_data, _nframes):
    max_per_joint = np.empty((_data.shape[0], _data.shape[2]))
    min_per_joint = np.empty((_data.shape[0], _data.shape[2]))
    for sidx in range(_data.shape[0]):
        max_per_joint[sidx, :] = np.amax(_data[sidx, :int(_nframes[sidx] - 1), :], axis=0)
        min_per_joint[sidx, :] = np.amin(_data[sidx, :int(_nframes[sidx] - 1), :], axis=0)
    max_per_joint = np.amax(max_per_joint, axis=0)
    min_per_joint = np.amin(min_per_joint, axis=0)
    data_scaled = np.empty_like(_data)
    for sidx in range(_data.shape[0]):
        max_repeated = np.repeat(np.expand_dims(max_per_joint, axis=0), _nframes[sidx] - 1, axis=0)
        min_repeated = np.repeat(np.expand_dims(min_per_joint, axis=0), _nframes[sidx] - 1, axis=0)
        data_scaled[sidx, :int(_nframes[sidx] - 1), :] =\
            np.nan_to_num(np.divide(_data[sidx, :int(_nframes[sidx] - 1), :] - min_repeated,
                                    max_repeated - min_repeated))
    return data_scaled, max_per_joint, min_per_joint


# descale generated data
def descale(data, data_max, data_min):
    data_descaled = data*(data_max-data_min)+data_min
    return data_descaled


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def to_multi_hot(labels, threshold=0.2):
    labels_out = np.zeros_like(labels)
    # labels_out[np.arange(len(labels)), labels.argwhere(1)] = 1
    hot_idx = np.argwhere(labels > threshold)
    labels_out[hot_idx[:, 0], hot_idx[:, 1]] = 1
    return labels_out


class TrainTestLoader(torch.utils.data.Dataset):

    def __init__(self, data, poses, differentials, affective_features, num_frames, labels, coords, num_labels):
        # data: N T V to N V T
        self.data = np.swapaxes(data, 2, 1)

        # poses
        self.poses = poses

        # differentials: N T V to N V T
        self.diff = np.swapaxes(differentials, 2, 1)

        # affective features
        self.affs = affective_features

        # number of frames
        self.num_frames = num_frames

        # labels
        # self.labels = tf.keras.utils.to_categorical(labels, num_classes)
        self.labels = []
        label_partitions = np.append([0], np.cumsum(num_labels))
        for lpidx in range(len(num_labels)):
            self.labels.append(to_multi_hot(labels[:, label_partitions[lpidx]:label_partitions[lpidx + 1]]))

        self.N, self.V, self.T = self.diff.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # data
        data = np.array(self.data[index])

        # poses
        poses = np.array(self.poses[index])

        # differentials
        differentials = np.array(self.diff[index])

        # affective features
        affs = np.array(self.affs[index])

        # number of frames
        num_frames = self.num_frames[index]

        # labels
        labels_list = []
        for lidx in range(len(self.labels)):
            labels_list.append(self.labels[lidx][index])

        # processing
        # if self.random_choose:
        #     data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
        #     data_numpy = tools.random_move(data_numpy)

        return data, poses, differentials, affs, num_frames, labels_list
