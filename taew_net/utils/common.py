import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import torch

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from torch.autograd import Variable
from utils.Quaternions import Quaternions


def fleiss_kappa(M):
    """
    See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
    :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of
    categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject
    to the `j`th category.
    :type M: numpy matrix
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators

    p = np.sum(M, axis=0) / (N * n_annotators)
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N
    PbarE = np.sum(p * p)

    chance_agreement = PbarE
    actual_above_chance = Pbar - PbarE
    max_above_chance = 1 - PbarE
    kappa = actual_above_chance / max_above_chance

    return chance_agreement, actual_above_chance, max_above_chance, kappa


def plot_features(features, labels, font_family='DejaVu Sans', font_size=30):
    pca = PCA(n_components=3)
    components = pca.fit_transform(features)
    data_viz = np.append(components, np.expand_dims(labels, axis=1), axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    targets = ['Happy', 'Angry', 'Neutral', 'Sad']
    colors = ['c', 'm', 'y', 'k']
    for i, color in enumerate(colors):
        label_idx = np.where(data_viz[:, -1] == i)
        ax.scatter(data_viz[label_idx, 0],
                   data_viz[label_idx, 1],
                   data_viz[label_idx, 2],
                   c=color, s=200, alpha=1)
    ax.legend(targets)
    ax.grid()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    font = {'font.family': font_family,
            'font.size': font_size}
    plt.rcParams.update(font)
    # fig_manager = plt.get_current_fig_manager()
    plt.savefig('deep_features_scatter.png')
    plt.show()


def to_multi_hot(labels, threshold=0.25):
    labels_out = np.zeros_like(labels)
    # labels_out[np.arange(len(labels)), labels.argwhere(1)] = 1
    hot_idx = np.argwhere(labels > threshold)
    labels_out[hot_idx[:, 0], hot_idx[:, 1]] = 1
    return labels_out


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between_vectors(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_points(p1, p2, p3):
    """ Returns the angle in radians between vectors 'p1' - 'p2' and 'p3' - 'p2'::
    """
    u1 = unit_vector(p1 - p2)
    u2 = unit_vector(p3 - p2)
    return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))


def dist_between(v1, v2):
    """ Returns the l2-norm distance between vectors 'v1' and 'v2'::
    """
    return np.linalg.norm(v1 - v2)


def area_of_triangle(v1, v2, v3):
    a = np.linalg.norm(v1 - v2)
    b = np.linalg.norm(v2 - v3)
    c = np.linalg.norm(v3 - v1)
    s = (a + b + c) / 2.
    return np.sqrt(np.abs(s * (s - a) * (s - b) * (s - c)))


def get_joints(d, g, s, t):
    if d == 'ewalk':
        root = g[s, t, 0, :]
        spine = g[s, t, 1, :]
        neck = g[s, t, 2, :]
        head = g[s, t, 3, :]
        left_shoulder = g[s, t, 4, :]
        left_elbow = g[s, t, 5, :]
        left_hand = g[s, t, 6, :]
        right_shoulder = g[s, t, 7, :]
        right_elbow = g[s, t, 8, :]
        right_hand = g[s, t, 9, :]
        left_hip = g[s, t, 10, :]
        left_knee = g[s, t, 11, :]
        left_foot = g[s, t, 12, :]
        right_hip = g[s, t, 13, :]
        right_knee = g[s, t, 14, :]
        right_foot = g[s, t, 15, :]
        return root, spine, neck, head, \
               left_shoulder, left_elbow, left_hand, \
               right_shoulder, right_elbow, right_hand, \
               left_hip, left_knee, left_foot, \
               right_hip, right_knee, right_foot
    elif d == 'edin':
        root = g[s, t, 0, :]
        left_hip = g[s, t, 1, :]
        left_knee = g[s, t, 2, :]
        left_heel = g[s, t, 3, :]
        left_toe = g[s, t, 4, :]
        right_hip = g[s, t, 5, :]
        right_knee = g[s, t, 6, :]
        right_heel = g[s, t, 7, :]
        right_toe = g[s, t, 8, :]
        lower_back = g[s, t, 9, :]
        spine = g[s, t, 10, :]
        neck = g[s, t, 11, :]
        head = g[s, t, 12, :]
        left_shoulder = g[s, t, 13, :]
        left_elbow = g[s, t, 14, :]
        left_hand = g[s, t, 15, :]
        left_hand_index = g[s, t, 16, :]
        right_shoulder = g[s, t, 17, :]
        right_elbow = g[s, t, 18, :]
        right_hand = g[s, t, 19, :]
        right_hand_index = g[s, t, 20, :]
        return root, left_hip, left_knee, left_heel, left_toe, \
               right_hip, right_knee, right_heel, right_toe, \
               lower_back, spine, neck, head, \
               left_shoulder, left_elbow, left_hand, left_hand_index, \
               right_shoulder, right_elbow, right_hand, right_hand_index


def get_rotation_matrix(axis_normalized, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    a = np.cos(theta / 2.0)
    bcd = -axis_normalized * np.sin(theta[:, None] / 2.0)
    b = bcd[:, 0]
    c = bcd[:, 1]
    d = bcd[:, 2]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def get_rotated_points(axis, theta, points):
    rotation_matrix = get_rotation_matrix(axis, theta)
    points = np.reshape(points, (-1, 3)).transpose()
    points_rotated = np.zeros_like(points)
    for idx in range(rotation_matrix.shape[-1]):
        points_rotated[:, idx] = rotation_matrix[:, :, idx] @ points[:, idx]
    return np.nan_to_num(points_rotated.transpose().flatten())


def get_del_orientation(pos1, pos2, dim):
    a = np.reshape(pos1 / np.linalg.norm(pos1), (-1, dim))
    b = np.reshape(pos2 / np.linalg.norm(pos1), (-1, dim))
    axis = np.cross(a, b)
    # axis_norm = np.sum(axis ** 2, axis=-1) ** 0.5
    # axis_normalized = axis / axis_norm[:, None]
    a_norm = np.sum(a ** 2, axis=-1) ** 0.5
    b_norm = np.sum(b ** 2, axis=-1) ** 0.5
    theta = np.arccos(np.divide(np.einsum('ij, ij->i', a, b), np.multiply(a_norm, b_norm)))
    # return np.nan_to_num(axis_normalized * theta[:, None]), axis_normalized, theta
    return axis, np.nan_to_num(theta)


def get_del_pos_and_orientation(pos1, pos2, dim):
    # del_orientation, axis, theta = get_del_orientation(pos1, pos2, dim)
    # pos1_rotated = get_rotated_points(axis, theta, pos1)
    axis, theta = get_del_orientation(pos1, pos2, dim)
    quats = Quaternions.from_angle_axis(theta, axis).qs
    axis_norm = np.sum(axis ** 2, axis=-1) ** 0.5
    axis_normalized = axis / axis_norm[:, None]
    pos1_rotated = get_rotated_points(axis_normalized, theta, pos1)
    del_pos = np.reshape(pos2 - pos1_rotated, (-1, dim))
    # return np.append(del_pos, del_orientation, axis=1).flatten()
    return quats.flatten(), del_pos.flatten()


def get_velocity(pos_curr, pos_prev):
    vel = pos_curr - pos_prev
    return np.append(vel, np.linalg.norm(vel))


def get_acceleration(vel_curr, vel_prev):
    return vel_curr - vel_prev


def get_jerk(acc_curr, acc_prev):
    return np.linalg.norm(acc_curr - acc_prev)


def get_dynamics(pos_curr, pos_prev, vel_prev=None, acc_prev=None):
    vel_curr = get_velocity(pos_curr, pos_prev)
    if vel_prev is None:
        return vel_curr
    acc_curr = get_acceleration(vel_curr[:-1], vel_prev)
    if acc_prev is None:
        return np.concatenate((vel_curr, acc_curr))
    jerk = get_jerk(acc_curr, acc_prev)
    return np.concatenate((vel_curr, [jerk]))


def get_ewalk_affective_features(pos, dim, affs_dim):
    # 0: root,              1: spine,       2: neck,        3: head,
    # 4: left_shoulder,     5: left_elbow,  6: left_hand,
    # 7: right_shoulder,    8: right_elbow, 9: right_hand,
    # 10: left_hip,         11: left_knee,  12: left_foot,
    # 13: right_hip,        14: right_knee, 15: right_foot

    up_vector = np.array([0, 1, 0])
    affective_features = np.zeros(affs_dim)
    fidx = 0

    pos_expanded = np.expand_dims(np.expand_dims(np.reshape(pos, (-1, dim)), axis=0), axis=0)
    _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15 = get_joints('ewalk', pos_expanded, 0, 0)

    affective_features[fidx] = angle_between_points(_7, _2, _4)
    fidx += 1
    affective_features[fidx] = angle_between_points(_7, _1, _4)
    fidx += 1
    affective_features[fidx] = angle_between_points(_7, _0, _4)
    fidx += 1
    affective_features[fidx] = angle_between_vectors(_1 - _0, up_vector)
    fidx += 1
    affective_features[fidx] = angle_between_vectors(_2 - _0, up_vector)
    fidx += 1
    affective_features[fidx] = angle_between_vectors(_3 - _0, up_vector)
    fidx += 1
    affective_features[fidx] = dist_between(_6, _2) / dist_between(_6, _0)
    fidx += 1
    affective_features[fidx] = dist_between(_9, _2) / dist_between(_9, _0)
    fidx += 1
    affective_features[fidx] = dist_between(_6, _2) / dist_between(_9, _2)
    fidx += 1
    affective_features[fidx] = dist_between(_6, _0) / dist_between(_9, _0)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_7, _2, _4) / area_of_triangle(_7, _0, _4)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_7, _1, _4) / area_of_triangle(_7, _0, _4)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_8, _2, _5) / area_of_triangle(_8, _0, _5)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_8, _1, _5) / area_of_triangle(_8, _0, _5)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_9, _2, _6) / area_of_triangle(_9, _0, _6)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_9, _1, _6) / area_of_triangle(_9, _0, _6)
    fidx += 1
    affective_features[fidx] = angle_between_points(_2, _4, _5)
    fidx += 1
    affective_features[fidx] = angle_between_points(_2, _7, _8)
    fidx += 1
    affective_features[fidx] = angle_between_points(_4, _5, _6)
    fidx += 1
    affective_features[fidx] = angle_between_points(_7, _8, _9)
    fidx += 1
    affective_features[fidx] = angle_between_points(_3, _2, _4)
    fidx += 1
    affective_features[fidx] = angle_between_points(_3, _2, _7)
    fidx += 1
    affective_features[fidx] = angle_between_points(_13, _0, _10)
    fidx += 1
    affective_features[fidx] = angle_between_points(_13, _1, _10)
    fidx += 1
    affective_features[fidx] = angle_between_points(_13, _2, _10)
    fidx += 1
    affective_features[fidx] = dist_between(_12, _2) / dist_between(_12, _0)
    fidx += 1
    affective_features[fidx] = dist_between(_15, _2) / dist_between(_15, _0)
    fidx += 1
    affective_features[fidx] = dist_between(_12, _2) / dist_between(_15, _2)
    fidx += 1
    affective_features[fidx] = dist_between(_12, _0) / dist_between(_15, _0)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_13, _0, _10) / area_of_triangle(_13, _2, _10)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_13, _0, _10) / area_of_triangle(_13, _1, _10)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_14, _0, _11) / area_of_triangle(_14, _2, _11)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_14, _0, _11) / area_of_triangle(_14, _1, _11)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_15, _0, _12) / area_of_triangle(_15, _1, _12)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_15, _0, _12) / area_of_triangle(_15, _2, _12)
    fidx += 1
    affective_features[fidx] = angle_between_points(_0, _10, _11)
    fidx += 1
    affective_features[fidx] = angle_between_points(_0, _13, _14)
    fidx += 1
    affective_features[fidx] = angle_between_points(_10, _11, _12)
    fidx += 1
    affective_features[fidx] = angle_between_points(_13, _14, _15)
    fidx += 1

    return np.nan_to_num(affective_features)


def get_edin_affective_features(pos, dim, affs_dim):
    # 0: root,              1: left_hip,        2: left_knee,   3: left_heel,           4: left_toe,
    # 5: right_hip,         6: right_knee,      7: right_heel,  8: right_toe,
    # 9: lower_back,        10: spine,          11: neck,       12: head,
    # 13: left_shoulder,    14: left_elbow,     15: left_hand,  16: left_hand_index,
    # 17: right_shoulder,   18: right_elbow,    19: right_hand, 20: right_hand_index

    # 14 features
    affective_features = np.zeros(affs_dim)
    fidx = 0

    pos_expanded = np.expand_dims(np.expand_dims(np.reshape(pos, (-1, dim)), axis=0), axis=0)
    _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,\
    _11, _12, _13, _14, _15, _16, _17, _18, _19, _20 = get_joints('edin', pos_expanded, 0, 0)

    affective_features[fidx] = angle_between_points(_17, _9, _13)
    fidx += 1
    affective_features[fidx] = angle_between_points(_19, _9, _15)
    fidx += 1
    affective_features[fidx] = dist_between(_16, _11) / dist_between(_16, _0)
    fidx += 1
    affective_features[fidx] = dist_between(_20, _11) / dist_between(_20, _0)
    fidx += 1
    affective_features[fidx] = dist_between(_16, _20) / dist_between(_11, _0)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_17, _9, _13) / area_of_triangle(_17, _0, _13)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_19, _9, _15) / area_of_triangle(_19, _0, _15)
    fidx += 1
    affective_features[fidx] = angle_between_points(_13, _14, _15)
    fidx += 1
    affective_features[fidx] = angle_between_points(_17, _18, _19)
    fidx += 1
    affective_features[fidx] = angle_between_points(_12, _11, _13)
    fidx += 1
    affective_features[fidx] = angle_between_points(_12, _11, _17)
    fidx += 1
    affective_features[fidx] = angle_between_points(_12, _0, _2)
    fidx += 1
    affective_features[fidx] = angle_between_points(_12, _0, _6)
    fidx += 1
    affective_features[fidx] = dist_between(_4, _8) / dist_between(_11, _0)
    fidx += 1
    affective_features[fidx] = angle_between_points(_8, _0, _4)
    fidx += 1
    affective_features[fidx] = angle_between_points(_1, _2, _3)
    fidx += 1
    affective_features[fidx] = angle_between_points(_5, _6, _7)
    fidx += 1
    affective_features[fidx] = area_of_triangle(_20, _11, _16) / area_of_triangle(_8, _0, _4)
    fidx += 1

    return np.nan_to_num(affective_features)


def get_vel_and_acc_idx(num_coords, coord_dim):
    vel_idx = np.arange(num_coords) + np.repeat(np.arange(0, num_coords, coord_dim), coord_dim)
    acc_idx = vel_idx + coord_dim
    return vel_idx, acc_idx


def get_ewalk_differentials_with_padding(gaits, num_frames, coords):
    dataset = 'ewalk'
    affs_dim = 39
    num_samples = gaits.shape[0]
    num_frames_max = gaits.shape[1]
    num_coords = gaits.shape[2]
    gaits_reshaped = np.reshape(gaits, (num_samples, num_frames_max, int(num_coords / coords), coords))
    poses = np.zeros((num_samples, num_coords))
    gaits_transformed = np.zeros((num_samples, num_frames_max, num_coords))
    differentials = np.zeros((num_samples, num_frames_max - 1, num_coords * 2))
    affective_features = np.zeros((num_samples, affs_dim))
    Y = np.array(get_joints(dataset, gaits_reshaped, 0, 0)).transpose()
    for sidx in range(num_samples):
        affective_features_curr = np.zeros((num_frames_max, affs_dim))
        X = np.array(get_joints(dataset, gaits_reshaped, sidx, 0)).transpose()
        R, _, t = get_transformation(X, Y)
        for tidx in range(int(num_frames[sidx])):
            Xtx = np.array(get_joints(dataset, gaits_reshaped, sidx, tidx)).transpose()
            gaits_transformed[sidx, tidx, :] = (np.dot(R, Xtx) + np.tile(
                np.reshape(t, (t.shape[0], 1)), (1, Xtx.shape[1]))).transpose().flatten()
            affective_features_curr[tidx, :] = \
                get_ewalk_affective_features(gaits_transformed[sidx, tidx, :], coords, affs_dim)
            if tidx == 0:
                poses[sidx, :] = gaits_transformed[sidx, tidx, :]
            else:
                differentials[sidx, tidx - 1, :] = \
                    get_del_pos_and_orientation(gaits_transformed[sidx, 0, :],
                                                gaits_transformed[sidx, tidx, :], coords)
        affective_features[sidx, :] = np.mean(affective_features_curr, axis=0)
    return poses, differentials, affective_features


def get_edin_differentials_with_padding(gaits, num_frames, coords):
    dataset = 'edin'
    affs_dim = 18
    num_samples = gaits.shape[0]
    num_frames_max = gaits.shape[1]
    num_coords = gaits.shape[2]
    num_joints = int(num_coords / coords)
    poses = np.zeros((num_samples, num_coords))
    rotations = np.zeros((num_samples, num_frames_max, num_joints * 4))
    translations = np.zeros((num_samples, num_frames_max, num_joints * 3))
    affective_features = np.zeros((num_samples, num_frames_max, affs_dim))
    for sidx in range(num_samples):
        for tidx in range(int(num_frames[sidx])):
            affective_features[sidx, tidx, :] = \
                get_edin_affective_features(gaits[sidx, tidx, :], coords, affs_dim)
            if tidx == 0:
                poses[sidx, :] = gaits[sidx, tidx, :]
                rotations[sidx, tidx, :] = Quaternions.id(num_joints).qs.flatten()
            else:
                rotations[sidx, tidx, :], translations[sidx, tidx, :] = \
                    get_del_pos_and_orientation(gaits[sidx, tidx - 1, :],
                                                gaits[sidx, tidx, :], coords)
    return poses, rotations, translations, affective_features


def get_joints_from_mocap_data(data, apply_transformations=True):
    data = np.moveaxis(np.squeeze(data), -1, 0)
    if not apply_transformations:
        joints = np.copy(data)
    else:
        if data.shape[-1] == 73:
            joints, root_x, root_z, root_r = data[:, 3:-7], data[:, -7], data[:, -6], data[:, -5]
        elif data.shape[-1] == 66:
            joints, root_x, root_z, root_r = data[:, :-3], data[:, -3], data[:, -2], data[:, -1]
    joints = joints.reshape((len(joints), -1, 3))

    rotations = np.empty((len(joints), 4))
    translations = np.empty((len(joints), 3))
    rotations[0, :] = Quaternions.id(1).qs
    offsets = []
    translations[0, :] = np.array([[0, 0, 0]])

    if apply_transformations:
        for i in range(len(joints)):
            joints[i, :, :] = Quaternions(rotations[i, :]) * joints[i]
            joints[i, :, 0] = joints[i, :, 0] + translations[i, 0]
            joints[i, :, 2] = joints[i, :, 2] + translations[i, 2]
            if i + 1 < len(joints):
                rotations[i + 1, :] = (Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0])) *
                                       Quaternions(rotations[i, :])).qs
                offsets.append(Quaternions(rotations[i + 1, :]) * np.array([0, 0, 1]))
                translations[i + 1, :] = translations[i, :] +\
                                         Quaternions(rotations[i + 1, :]) * np.array([root_x[i], 0, root_z[i]])
    return joints, rotations, translations


def reconstruct_gait(pose, differentials, V):
    num_frames = differentials.shape[1]
    gait = np.zeros((num_frames + 1, pose.shape[0]))
    gait[0, :] = pose
    if differentials.shape[-1] == V * 7:
        del_pos_available = True
    else:
        del_pos_available = False
    for tidx in range(num_frames):
        if del_pos_available:
            differentials_curr = np.reshape(differentials[:, tidx, :], (-1, 7))
            del_pos = differentials_curr[:, 4:].flatten()
        else:
            differentials_curr = np.reshape(differentials[:, tidx, :], (-1, 4))
            del_pos = np.zeros((differentials_curr.shape[0] * 3))
        del_orientation = differentials_curr[:, :4]
        theta, axis_normalized = Quaternions.angle_axis(Quaternions(del_orientation))
        theta = np.remainder(theta + np.pi, 2 * np.pi) - np.pi
        axis_normalized = axis_normalized / np.linalg.norm(axis_normalized, axis=1)[:, None]
        # theta = np.linalg.norm(del_orientation, axis=1)
        # axis_normalized = del_orientation / theta[:, None]
        gait[tidx + 1, :] = get_rotated_points(axis_normalized, theta, gait[tidx, :]) + del_pos
        # gait[tidx + 1, :] = gait[tidx, :]
    return gait


def get_transformation(X, Y):
    """
    Args:
        X: k x n source shape
        Y: k x n destination shape such that Y[:, i] is the correspondence of X[:, i]
    Returns: rotation R, scaling c, translation t such that ||Y - (cRX+t)||_2_1 is minimized.
    """
    """
    Copyright: Carlo Nicolini, 2013
    Code adapted from the Mark Paskin Matlab version
    from http://openslam.informatik.uni-freiburg.de/data/svn/tjtf/trunk/matlab/ralign.m 
    """

    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc = X - np.tile(mx, (n, 1)).T
    Yc = Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc * Xc, 0))
    sy = np.mean(np.sum(Yc * Yc, 0))

    M = np.dot(Yc, Xc.T) / n

    U, D, V = np.linalg.svd(M, full_matrices=True, compute_uv=True)
    V = V.T.copy()
    # print U,"\n\n",D,"\n\n",V
    r = np.rank(M)
    d = np.linalg.det(M)
    S = np.eye(m)
    if r > (m - 1):
        if np.det(M) < 0:
            S[m, m] = -1
        elif r == m - 1:
            if np.det(U) * np.det(V) < 0:
                S[m, m] = -1
        else:
            R = np.eye(2)
            c = 1
            t = np.zeros(2)
            return R, c, t

    R = np.dot(np.dot(U, S), V.T)
    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)

    return R, c, t


def fit_sin(tt, yy):
    '''
    Fit sin to the input time sequence, and return fitting parameters
    "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
    '''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w * t + p) + c

    popt, pcov = opt.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w / (2. * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. / f, "fitfunc": fitfunc,
            "maxcov": np.max(pcov), "rawres": (guess, popt, pcov)}
