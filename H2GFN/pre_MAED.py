import torch.nn as nn
from GAT import GAT
import torch.nn.functional as F
import numpy as np
import torch
import dgl
import networkx as nx

def kNN_matrix(matrix, k):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)

    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)


def Similarity_graph(data, args):
    didi_matrix = kNN_matrix(data['diss'], args.neighbor)
    mimi_matrix = kNN_matrix(data['mis'], args.neighbor)

    didi_nx = nx.from_numpy_array(didi_matrix)
    mimi_nx = nx.from_numpy_array(mimi_matrix)

    didi_graph = dgl.from_networkx(didi_nx)
    mimi_graph = dgl.from_networkx(mimi_nx)

    didi_graph = dgl.add_self_loop(didi_graph)
    mimi_graph = dgl.add_self_loop(mimi_graph)

    return didi_graph, mimi_graph

def dysce(x, y, alpha=2):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

class pre_MAED(nn.Module):
    def __init__(self, args, in_dim):
        super(pre_MAED, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.maed_layers = self.args.maed_layers
        self.maed_hid = self.args.maed_hid
        self.maed_heads = self.args.maed_heads
        self.maed_out = self.args.maed_out
        self.r_mask = self.args.r_mask
        self.r_replace = self.args.r_replace
        self.r_token = 1 - self.r_replace
        self.enc_token = nn.Parameter(torch.zeros(1, self.in_dim))
        self.dysce_loss = dysce

        enc_sghid = self.maed_hid // self.maed_heads
        dec_in_dim = self.maed_hid
        dec_sghid = self.maed_hid // self.maed_out

        self.enc_dec = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.encoder = GAT(self.args, self.in_dim, enc_sghid, enc_sghid, self.maed_layers, self.maed_heads,
                           encoding=True)
        self.decoder = GAT(self.args, dec_in_dim, self.in_dim, dec_sghid, num_layers=1, num_outheads=1, encoding=False)

    def forward(self, G, X, alpha=2.0):
        loss, final_X = self.reconstruct_masked_features(G, X, alpha)
        loss_item = {"loss": loss.item()}
        return loss, loss_item, final_X

    def reconstruct_masked_features(self, G, X, alpha):
        num_nodes = G.num_nodes()
        perm = torch.randperm(num_nodes, device=X.device)
        num_mask_nodes = int(self.r_mask * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = X.clone()
        if self.r_replace > 0:
            num_noise_nodes = int(self.r_replace * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=X.device)
            token_nodes = mask_nodes[perm_mask[: int(self.r_token * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
            noise_chosen = torch.randperm(num_nodes, device=X.device)[:num_noise_nodes]

            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = X[noise_chosen]
        else:
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_token

        r_drop = getattr(self.args, "r_drop", 0.0)
        if self.training and r_drop > 0:
            num_edges = G.num_edges()
            perm_edges = torch.randperm(num_edges, device=X.device)
            keep_cnt = int(num_edges * (1 - r_drop))
            keep_edges = perm_edges[:keep_cnt]
            new_G = dgl.edge_subgraph(G, keep_edges, relabel_nodes=False)
            new_G = dgl.remove_self_loop(new_G)
            new_G = dgl.add_self_loop(new_G)
        else:
            new_G = G

        enc_Henc, _ = self.encoder(new_G, out_x, return_hidden=True)
        Henc = self.enc_dec(enc_Henc)
        Henc[mask_nodes] = 0
        final_X = self.decoder(G, Henc)
        init_Xmask = X[mask_nodes]
        final_Xmask = final_X[mask_nodes]
        loss = self.dysce_loss(final_Xmask, init_Xmask, alpha=alpha)
        return loss, final_X

    def encoding_maed(self, G, X):
        new_X = self.encoder(G, X)
        return new_X