import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

from torch.autograd import Variable
from utils.common import *
from utils.Quaternions_torch import qeuler, euler_to_quaternion

torch.manual_seed(1234)


class HAPPY(nn.Module):

    def __init__(self, dataset, T, V, C, A, T_out, num_classes, sizes=[96, 64, 48, 16, 8],
                 **kwargs):

        super().__init__()

        self.T = T
        self.V = V
        self.C = C
        self.A = A
        self.T_out = T_out
        if dataset == 'edin':
            self.encoder = Encoder_Edin(T, V, C, A, num_classes, sizes)
            self.decoder = Decoder_Edin(T, V, C, A, T_out, num_classes, sizes)

    def forward(self, x_poses, x_diffs, x_affs, h_enc=None, h_dec1=None, h_dec=None, teacher_steps=0):

        affs_pred, labels_pred, h_enc = self.encoder(x_poses, x_diffs, x_affs, h_enc)

        x_recons, h_dec1, h_dec, x_recons_pre_norm = self.decoder(affs_pred, h_dec1, h_dec,
                                                                  teacher_seq=x_diffs,
                                                                  teacher_steps=teacher_steps)

        return labels_pred, x_recons, x_recons_pre_norm, affs_pred, h_enc, h_dec1, h_dec

    def inference(self, n=1, ldec=None):

        batch_size = n
        z = to_var(torch.randn([batch_size, self.A]))

        recon_x = self.decoder(z, ldec)

        return recon_x


class Encoder_Edin(nn.Module):

    def __init__(self, T, V, C, A, num_classes, sizes, dropout_factor=0.1, **kwargs):
        super().__init__()

        self.T = T
        self.V = V
        self.C = C
        self.A = A
        self.sizes = sizes
        self.num_classes = num_classes
        self.GRU_num_layers = 2
        self.activation = nn.ELU()

        # build networks
        self.BatchNorm_layer1 = nn.BatchNorm1d(T - 1)
        self.h0 = nn.Parameter(torch.zeros(self.GRU_num_layers, 1, self.V * sizes[0]).normal_(std=0.0),
                               requires_grad=True).cuda()
        self.GRU_layer1 = nn.GRU(self.V * self.C, self.V * sizes[0],
                                 num_layers=self.GRU_num_layers, batch_first=True)
        self.BatchNorm_layer2 = nn.ModuleList()
        self.Linear_layer2 = nn.ModuleList()
        for lidx in range(V):
            self.BatchNorm_layer2.append(nn.BatchNorm1d(sizes[0]))
            self.Linear_layer2.append(nn.Conv1d(sizes[0], sizes[1], 5, padding=2))
        self.BatchNorm_layer3 = nn.ModuleList((
            nn.BatchNorm1d(sizes[1]),
            nn.BatchNorm1d(sizes[1]),
            nn.BatchNorm1d(sizes[1]),
            nn.BatchNorm1d(sizes[1]),
            nn.BatchNorm1d(sizes[1]),
            nn.BatchNorm1d(sizes[2]),
            nn.BatchNorm1d(sizes[2])
        ))
        self.Linear_layer3 = nn.ModuleList((
            nn.Conv1d(sizes[1], sizes[2], 3, padding=1),
            nn.Conv1d(sizes[1], sizes[2], 3, padding=1),
            nn.Conv1d(sizes[1], sizes[2], 3, padding=1),
            nn.Conv1d(sizes[1], sizes[2], 3, padding=1),
            nn.Conv1d(sizes[1], sizes[2], 3, padding=1),
            nn.Conv1d(sizes[2], sizes[2], 3, padding=1),
            nn.Conv1d(sizes[2], sizes[2], 3, padding=1)
        ))

        # fcn for encoding
        self.z_to_aff = nn.Conv1d(sizes[2], A, 3, padding=1)

        # fcn for classifying
        self.BatchNorm_layer4 = nn.ModuleList()
        self.Linear_layer4 = nn.ModuleList()
        self.Dropout_layer4 = nn.ModuleList()

        c = num_classes[0]
        self.BatchNorm_layer4.append(nn.BatchNorm1d(A))
        self.Linear_layer4.append(nn.Conv1d(A, sizes[3], 1))
        self.Dropout_layer4.append(nn.Dropout(dropout_factor))
        self.BatchNorm_layer5 = nn.ModuleList()
        self.Linear_layer5 = nn.ModuleList()
        self.Dropout_layer5 = nn.ModuleList()

        self.Linear_layer5.append(nn.Linear(sizes[3], sizes[4]))
        self.Dropout_layer5.append(nn.Dropout(dropout_factor))
        self.Classifiers = nn.ModuleList()
        self.Classifiers_ip = nn.ModuleList()

        self.Classifiers.append(nn.Linear(sizes[-1], c))
        self.Classifiers_ip.append(nn.Linear(sizes[-1], c))
        self.BatchNorm_layer6 = nn.BatchNorm1d(T - 1)
        self.Linear_layer7 = nn.Linear(self.T - 1, 1)
        self.to_probs = nn.Sigmoid()

    def create_joints_list(self, x_diffs):
        x_diffs_list = []
        for vidx in range(0, x_diffs.shape[1], self.sizes[0]):
            x_diffs_list.append(x_diffs[:, vidx:vidx + self.sizes[0], :].permute(0, 2, 1))
        return x_diffs_list

    def forward(self, x_pose, x_diffs, x_affs, h):

        x_diffs = x_diffs.permute(0, 2, 1)
        x_e = qeuler(x_diffs.reshape(x_diffs.shape[0], x_diffs.shape[1], -1, 4).contiguous(), order='zyx')
        x_e = F.normalize(x_e).reshape(x_diffs.shape[0], x_diffs.shape[1], -1)
        if h is None:
            h_in = self.h0.expand(-1, x_diffs.shape[0], -1).contiguous()
        else:
            h_in = h.clone()
        x_diffs, h = self.GRU_layer1(x_e, h_in)
        max_hidden_diff = torch.max(torch.abs(h[-1, :, :] - h_in[-1, :, :]))
        if max_hidden_diff < 1e-2:
            print('Warning! Max hidden out diff is {}'.format(max_hidden_diff))
        x_diffs_list = self.create_joints_list(x_diffs.permute(0, 2, 1))
        for lidx, (Linear, BatchNorm) in enumerate(zip(self.Linear_layer2, self.BatchNorm_layer2)):
            x_diffs_list[lidx] = self.activation(Linear(x_diffs_list[lidx].permute(0, 2, 1)))

        last_dim = 3
        rt_lb_sp_n_hd = self.activation(self.Linear_layer3[0](
            torch.sum(torch.cat((
                0.1 * x_diffs_list[0].unsqueeze(-1),
                0.1 * x_diffs_list[9].unsqueeze(-1),
                0.2 * x_diffs_list[10].unsqueeze(-1),
                0.2 * x_diffs_list[11].unsqueeze(-1),
                0.4 * x_diffs_list[12].unsqueeze(-1)), dim=last_dim), dim=last_dim)))

        l_sh_e_hn_hx = self.activation(self.Linear_layer3[1](
            torch.sum(torch.cat((
                0.1 * x_diffs_list[13].unsqueeze(-1),
                0.2 * x_diffs_list[14].unsqueeze(-1),
                0.3 * x_diffs_list[15].unsqueeze(-1),
                0.4 * x_diffs_list[16].unsqueeze(-1)), dim=last_dim), dim=last_dim)))

        r_sh_e_hn_hx = self.activation(self.Linear_layer3[2](
            torch.sum(torch.cat((
                0.1 * x_diffs_list[17].unsqueeze(-1),
                0.2 * x_diffs_list[18].unsqueeze(-1),
                0.3 * x_diffs_list[19].unsqueeze(-1),
                0.4 * x_diffs_list[20].unsqueeze(-1)), dim=last_dim), dim=last_dim)))

        l_hp_kn_hl_t = self.activation(self.Linear_layer3[3](
            torch.sum(torch.cat((
                0.1 * x_diffs_list[1].unsqueeze(-1),
                0.2 * x_diffs_list[2].unsqueeze(-1),
                0.3 * x_diffs_list[3].unsqueeze(-1),
                0.4 *x_diffs_list[4].unsqueeze(-1)), dim=last_dim), dim=last_dim)))

        r_hp_kn_hl_t = self.activation(self.Linear_layer3[4](
            torch.sum(torch.cat((
                0.1 * x_diffs_list[5].unsqueeze(-1),
                0.2 * x_diffs_list[6].unsqueeze(-1),
                0.3 * x_diffs_list[7].unsqueeze(-1),
                0.4 * x_diffs_list[8].unsqueeze(-1)), dim=last_dim), dim=last_dim)))

        z = self.activation(self.Linear_layer3[5](
            torch.sum(torch.cat((
                0.12 * rt_lb_sp_n_hd.unsqueeze(-1),
                0.22 * l_sh_e_hn_hx.unsqueeze(-1),
                0.22 * r_sh_e_hn_hx.unsqueeze(-1),
                0.22 * l_hp_kn_hl_t.unsqueeze(-1),
                0.22 * r_hp_kn_hl_t.unsqueeze(-1)), dim=last_dim), dim=last_dim)))

        affs_pred = self.activation(self.z_to_aff(z))

        labels_pred = []
        cidx = 0
        labels_pred.append(self.activation(
            self.Dropout_layer4[cidx](
                self.Linear_layer4[cidx](
                    affs_pred))))
        labels_pred[cidx] = self.activation(
            self.Dropout_layer5[cidx](
                self.Linear_layer5[cidx](
                    labels_pred[cidx].permute(0, 2, 1))))
        labels_pred[cidx] = self.BatchNorm_layer6(labels_pred[cidx])
        labels_pred[cidx] = self.Classifiers[cidx](labels_pred[cidx])
        labels_mean = torch.mean(labels_pred[cidx], dim=1)
        labels_pred[cidx] = self.to_probs(labels_pred[cidx])
        max_label_diff = torch.max(torch.abs(labels_pred[cidx] - labels_pred[cidx][np.roll(np.arange(labels_mean.shape[0]), -1), :, :]))
        if labels_pred[cidx].shape[0] > 1 and max_label_diff < 1e-3:
            print('Warning! Max label diff is {}'.format(max_label_diff))
        return affs_pred.permute(0, 2, 1), labels_pred, h


class Decoder_Edin(nn.Module):

    def __init__(self, T, V, C, A, T_out, num_classes, sizes, **kwargs):
        super().__init__()

        self.T = T
        self.V = V
        self.C = C + 1
        self.A = A
        self.T_out = T_out
        self.num_classes = num_classes
        self.sizes = sizes
        self.GRU_num_layers = 2
        self.activation = nn.ELU()

        # build networks
        self.Linear_layer4 = nn.Linear(A, sizes[2])
        self.BatchNorm_layer4 = nn.BatchNorm1d(sizes[2])
        self.Linear_layer3 = nn.ModuleList((
            nn.Linear(sizes[2], sizes[2]),
            nn.Linear(sizes[2], sizes[2]),
            nn.Linear(sizes[2], sizes[1]),
            nn.Linear(sizes[2], sizes[1]),
            nn.Linear(sizes[2], sizes[1]),
            nn.Linear(sizes[2], sizes[1]),
            nn.Linear(sizes[2], sizes[1])
        ))
        self.BatchNorm_layer3 = nn.ModuleList((
            nn.BatchNorm1d(sizes[2]),
            nn.BatchNorm1d(sizes[2]),
            nn.BatchNorm1d(sizes[1]),
            nn.BatchNorm1d(sizes[1]),
            nn.BatchNorm1d(sizes[1]),
            nn.BatchNorm1d(sizes[1]),
            nn.BatchNorm1d(sizes[1])
        ))
        self.Linear_layer2 = nn.ModuleList()
        self.BatchNorm_layer2 = nn.ModuleList()
        for lidx in range(V):
            self.Linear_layer2.append(nn.Linear(sizes[1], sizes[0]))
            self.BatchNorm_layer2.append(nn.BatchNorm1d(sizes[0]))
        self.GRU_layer1 = nn.GRU(self.V * sizes[0], self.V * self.C,
                                 num_layers=self.T_out, batch_first=True)
        self.h01 = nn.Parameter(torch.zeros(self.T_out, 1, self.V * self.C).normal_(std=0.01),
                                requires_grad=True).cuda()
        self.BatchNorm_layer1 = nn.BatchNorm1d(self.T_out)
        self.BatchNorm_all_layer1 = nn.BatchNorm1d(self.GRU_num_layers)
        self.GRU_layer0 = nn.GRU(self.V * self.C, self.V * self.C,
                                 num_layers=self.GRU_num_layers, batch_first=True)
        self.h00 = nn.Parameter(torch.zeros(self.GRU_num_layers, 1, self.V * self.C).normal_(std=0.01),
                                requires_grad=True).cuda()

    def concatenate_joints_list(self, x_diffs_list):
        x_diffs = x_diffs_list[0]
        for vidx in range(1, len(x_diffs_list)):
            x_diffs = torch.cat((x_diffs, x_diffs_list[vidx]), dim=-1)
        return x_diffs.unsqueeze(1)

    def forward(self, z, h1, h, teacher_seq=None, teacher_steps=0):

        z = self.Linear_layer4(z[:, 0, :])

        rt_lb_sp_n_hd = self.activation(self.Linear_layer3[1](z))
        l_hp_kn_hl_t = torch.zeros_like(rt_lb_sp_n_hd)
        l_hp_kn_hl_t.data = rt_lb_sp_n_hd.clone() * 0.22
        r_hp_kn_hl_t = torch.zeros_like(rt_lb_sp_n_hd)
        rt_lb_sp_n_hd.data = rt_lb_sp_n_hd.clone() * 0.22
        l_sh_e_hn_hx = torch.zeros_like(rt_lb_sp_n_hd)
        l_sh_e_hn_hx.data = rt_lb_sp_n_hd.clone() * 0.22
        r_sh_e_hn_hx = torch.zeros_like(rt_lb_sp_n_hd)
        r_sh_e_hn_hx.data = rt_lb_sp_n_hd.clone() * 0.22
        rt_lb_sp_n_hd = rt_lb_sp_n_hd * 0.12

        l_hp_kn_hl_t = self.activation(self.Linear_layer3[2](l_hp_kn_hl_t))
        r_hp_kn_hl_t = self.activation(self.Linear_layer3[3](r_hp_kn_hl_t))

        l_sh_e_hn_hx = self.activation(self.Linear_layer3[4](l_sh_e_hn_hx))
        r_sh_e_hn_hx = self.activation(self.Linear_layer3[5](r_sh_e_hn_hx))

        rt_lb_sp_n_hd = self.activation(self.Linear_layer3[6](rt_lb_sp_n_hd))

        root = torch.zeros_like(rt_lb_sp_n_hd)
        root.data = rt_lb_sp_n_hd.clone() * 0.1
        lwr_bk = torch.zeros_like(rt_lb_sp_n_hd)
        lwr_bk.data = rt_lb_sp_n_hd.clone() * 0.1
        spine = torch.zeros_like(rt_lb_sp_n_hd)
        spine.data = rt_lb_sp_n_hd.clone() * 0.2
        neck = torch.zeros_like(rt_lb_sp_n_hd)
        neck.data = rt_lb_sp_n_hd.clone() * 0.2
        head = torch.zeros_like(rt_lb_sp_n_hd)
        head.data = rt_lb_sp_n_hd.clone() * 0.4

        l_sh = torch.zeros_like(l_sh_e_hn_hx)
        l_sh.data = l_sh_e_hn_hx.clone() * 0.1
        l_el = torch.zeros_like(l_sh_e_hn_hx)
        l_el.data = l_sh_e_hn_hx.clone() * 0.2
        l_hn = torch.zeros_like(l_sh_e_hn_hx)
        l_hn.data = l_sh_e_hn_hx.clone() * 0.3
        l_hx = torch.zeros_like(l_sh_e_hn_hx)
        l_hx.data = l_sh_e_hn_hx.clone() * 0.4

        r_sh = torch.zeros_like(r_sh_e_hn_hx)
        r_sh.data = r_sh_e_hn_hx.clone() * 0.1
        r_el = torch.zeros_like(r_sh_e_hn_hx)
        r_el.data = r_sh_e_hn_hx.clone() * 0.2
        r_hn = torch.zeros_like(r_sh_e_hn_hx)
        r_hn.data = r_sh_e_hn_hx.clone() * 0.3
        r_hx = torch.zeros_like(r_sh_e_hn_hx)
        r_hx.data = r_sh_e_hn_hx.clone() * 0.4

        l_hp = torch.zeros_like(l_hp_kn_hl_t)
        l_hp.data = l_hp_kn_hl_t.clone() * 0.1
        l_kn = torch.zeros_like(l_hp_kn_hl_t)
        l_kn.data = l_hp_kn_hl_t.clone() * 0.2
        l_hl = torch.zeros_like(l_hp_kn_hl_t)
        l_hl.data = l_hp_kn_hl_t.clone() * 0.3
        l_t = torch.zeros_like(l_hp_kn_hl_t)
        l_t.data = l_hp_kn_hl_t.clone() * 0.4

        r_hp = torch.zeros_like(r_hp_kn_hl_t)
        r_hp.data = r_hp_kn_hl_t.clone() * 0.1
        r_kn = torch.zeros_like(r_hp_kn_hl_t)
        r_kn.data = r_hp_kn_hl_t.clone() * 0.2
        r_hl = torch.zeros_like(r_hp_kn_hl_t)
        r_hl.data = r_hp_kn_hl_t.clone() * 0.3
        r_t = torch.zeros_like(r_hp_kn_hl_t)
        r_t.data = r_hp_kn_hl_t.clone() * 0.4

        x_diffs_list = []
        x_diffs_list.append(root)
        x_diffs_list.append(l_hp)
        x_diffs_list.append(l_kn)
        x_diffs_list.append(l_hl)
        x_diffs_list.append(l_t)
        x_diffs_list.append(r_hp)
        x_diffs_list.append(r_kn)
        x_diffs_list.append(r_hl)
        x_diffs_list.append(r_t)
        x_diffs_list.append(lwr_bk)
        x_diffs_list.append(spine)
        x_diffs_list.append(neck)
        x_diffs_list.append(head)
        x_diffs_list.append(l_sh)
        x_diffs_list.append(l_el)
        x_diffs_list.append(l_hn)
        x_diffs_list.append(l_hx)
        x_diffs_list.append(r_sh)
        x_diffs_list.append(r_el)
        x_diffs_list.append(r_hn)
        x_diffs_list.append(r_hx)

        for lidx, (Linear, BatchNorm) in enumerate(zip(self.Linear_layer2, self.BatchNorm_layer2)):
            x_diffs_list[lidx] = Linear(x_diffs_list[lidx])
        x_diffs = self.concatenate_joints_list(x_diffs_list)

        if h1 is None:
            h1 = self.h01.expand(-1, x_diffs.shape[0], -1).contiguous()
        _, h1 = self.GRU_layer1(x_diffs, h1)
        x_diffs = torch.zeros((h1.shape[1], self.T_out, h1.shape[2])).cuda().float()
        x_diffs[:, :teacher_steps, :] = teacher_seq.permute(0, 2, 1)[:, :teacher_steps, :]
        x_diffs[:, teacher_steps:, :] = h1.permute(1, 0, 2)[:, teacher_steps:, :]
        x_diffs_out = torch.zeros((h1.shape[1], self.T_out, h1.shape[2])).cuda().float()
        if h is None:
            h = self.h00.expand(-1, x_diffs.shape[0], -1).contiguous()
        for t in range(self.T_out):
            out, h = self.GRU_layer0(x_diffs[:, t, :].unsqueeze(1), h)
            x_diffs_out[:, t, :] = out.squeeze()
        x_diffs_out = x_diffs_out.contiguous().view(x_diffs_out.shape[0], self.T_out, self.V, self.C)
        x_diffs_pre_norm = torch.zeros_like(x_diffs_out)
        x_diffs_pre_norm.data = x_diffs_out.clone()
        x_diffs_out = nn.functional.normalize(x_diffs_out, p=2, dim=3)
        x_diffs_out = x_diffs_out.reshape(-1, self.T_out, self.V * self.C)
        return x_diffs_out, h1, h, x_diffs_pre_norm
