from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Any
import torch
import numpy as np
from torch import Tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x

def constructNet(association_matrix):
    n, m = association_matrix.shape
    print("n:", n, "m:", m)
    matrix1 = torch.zeros((n, n), dtype=torch.int8)
    matrix2 = torch.zeros((m, m), dtype=torch.int8)
    mat1 = torch.cat((matrix1, association_matrix), dim=1)
    mat2 = torch.cat((association_matrix.t(), matrix2), dim=1)
    adj_0 = torch.cat((mat1, mat2), dim=0)
    return adj_0

def laplacian_positional_encoding(adj, pe_dim):
    adj = adj.float()
    N = torch.diag(torch.pow(torch.sum(adj, dim=1).clamp(min=1), -0.5))
    L = torch.eye(adj.shape[0]).to(device) - N @ adj @ N
    EigVal, EigVec = torch.linalg.eig(L)
    EigVal = EigVal.real
    EigVec = EigVec.real
    sorted_indices = EigVal.argsort()
    EigVec_sorted = EigVec[:, sorted_indices]
    lap_pos_enc = (EigVec_sorted[:, 1:int(pe_dim) + 1]).float()
    return lap_pos_enc

def re_features(adj, features, K):
    adj=adj.double()
    nodes_features = torch.empty(features.shape[0], 1, int(K+1), features.shape[1])
    for i in range(features.shape[0]):
        nodes_features[i, 0, 0, :] = features[i]
    x = features + torch.zeros_like(features)
    x = x.double()
    for i in range(int(K)):
        x = torch.matmul(adj, x)
        for index in range(features.shape[0]):
            nodes_features[index, 0, i + 1, :] = x[index]
    nodes_features = nodes_features.squeeze()
    return nodes_features

class PolynomialDecayLR(_LRScheduler):
    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False

def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)

def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1
    return link_labels

def gaussiansimilarity(interaction, nd, nm):
    norm_sq = np.linalg.norm(interaction)**2
    gamad = nd / norm_sq
    gamam = nm / norm_sq
    D = np.dot(interaction, interaction.T)
    d_diag = np.diag(D).reshape(-1, 1)
    d_dist_sq = d_diag + d_diag.T - 2 * D
    kd = np.exp(-gamad * d_dist_sq)
    E = np.dot(interaction.T, interaction)
    e_diag = np.diag(E).reshape(-1, 1)
    e_dist_sq = e_diag + e_diag.T - 2 * E
    km = np.exp(-gamam * e_dist_sq)
    return kd, km

def integratedsimilarity(mirnaS, mirnaSP, disS, disSP, kd, km):
    sm = mirnaS*mirnaSP+km*(-(mirnaSP-1))
    sd = disS*disSP+kd*(-(disSP-1))
    return sd, sm