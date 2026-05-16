import torch
import torch.nn as nn
from torch_geometric.nn import conv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_inf(adj, L, c):
    W = torch.zeros(adj.shape)
    D = (torch.diag(adj.sum(dim=1)) + 1e-15) ** (-1)
    for gamma in range(L):
        W += c * ((1 - c) ** gamma) * ((D * adj) ** gamma)
    W = torch.mul(W, adj > 0)
    W = torch.softmax(W, dim=1)
    return W


class GatingUnit(nn.Module):
    def __init__(self, dim):
        super(GatingUnit, self).__init__()
        self.gate_fc = nn.Linear(dim, dim)
        self.temp = nn.Parameter(torch.ones(1))
        self.res_scale = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, base_feat, high_order_feat):
        gate_score = self.gate_fc(high_order_feat)
        gate_weight = torch.sigmoid(gate_score / (self.temp + 1e-6))
        filtered_feat = high_order_feat * gate_weight
        return base_feat + self.res_scale * filtered_feat


class GMSTL(nn.Module):
    def __init__(self,
                 muti_in_dim1,
                 muti_in_dim2,
                 num_dis,
                 num_mirna,
                 layer1_hidden_units,
                 layer2_hidden_units,
                 threshold,
                 rwr_c,
                 gcn1_innum,
                 ):
        super().__init__()
        self.muti_in_dim1 = muti_in_dim1
        self.muti_in_dim2 = muti_in_dim2
        self.dis_nums, self.mirna_nums = num_dis, num_mirna
        self.threshold = threshold
        self.rwr_c = rwr_c
        self.gcn1_units, self.gcn2_units = layer1_hidden_units, layer2_hidden_units
        self.gcn1_innum = gcn1_innum

        # Scale L=2 GCNs
        self.gcn1 = conv.GCNConv(self.gcn1_innum, self.gcn1_units)
        self.gcn2 = conv.GCNConv(self.gcn1_units, self.gcn2_units)
        # Scale L=1 GCNs
        self.gcn3 = conv.GCNConv(self.gcn1_innum, self.gcn1_units)
        self.gcn4 = conv.GCNConv(self.gcn1_units, self.gcn2_units)
        # Scale L=0 GCNs
        self.gcn5 = conv.GCNConv(self.gcn1_innum, self.gcn1_units)
        self.gcn6 = conv.GCNConv(self.gcn1_units, self.gcn2_units)

        # Gate units for multi-scale feature integration
        feature_dim = self.gcn1_units + self.gcn2_units
        self.gate_l1 = GatingUnit(feature_dim)
        self.gate_l2 = GatingUnit(feature_dim)

        # Fusion layer
        self.cnn = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), stride=1, dilation=1, padding=1)

        self.relu = nn.ReLU()
        self.fc1 = nn.Sequential(
            nn.Linear(self.muti_in_dim1, self.gcn1_innum),
            nn.ReLU()
        )
        nn.init.xavier_normal_(self.fc1[0].weight, nn.init.calculate_gain('relu'))

    def forward(self, X, adj):
        X = self.fc1(X.to(torch.float32))

        # --- Scale L=0: Basic Topology ---
        idx_l0 = torch.nonzero(adj > self.threshold).to(torch.long)
        weight_l0 = adj[idx_l0[:, 0], idx_l0[:, 1]]
        emb5 = self.relu(self.gcn5(X, idx_l0.T, weight_l0))
        emb6 = self.relu(self.gcn6(emb5, idx_l0.T, weight_l0))
        emb_l0 = torch.cat((emb5, emb6), dim=1)

        adj_inf1 = compute_inf(adj.cpu(), L=1, c=self.rwr_c).to(device)
        idx_l1 = torch.nonzero(adj_inf1 > self.threshold).to(torch.long)
        weight_l1 = adj[idx_l1[:, 0], idx_l1[:, 1]]
        emb3 = self.relu(self.gcn3(X, idx_l1.T, weight_l1))
        emb4 = self.relu(self.gcn4(emb3, idx_l1.T, weight_l1))
        emb_l1_raw = torch.cat((emb3, emb4), dim=1)
        emb_l1 = self.gate_l1(emb_l0, emb_l1_raw)

        adj_inf2 = compute_inf(adj.cpu(), L=2, c=self.rwr_c).to(device)
        idx_l2 = torch.nonzero(adj_inf2 > self.threshold).to(torch.long)
        weight_l2 = adj[idx_l2[:, 0], idx_l2[:, 1]]
        emb1 = self.relu(self.gcn1(X, idx_l2.T, weight_l2))
        emb2 = self.relu(self.gcn2(emb1, idx_l2.T, weight_l2))
        emb_l2_raw = torch.cat((emb1, emb2), dim=1)
        emb_l2 = self.gate_l2(emb_l0, emb_l2_raw)

        emb_stack = torch.stack((emb_l0, emb_l1, emb_l2), dim=0).unsqueeze(0)
        emb_cnn = self.cnn(emb_stack)
        emb_all = emb_cnn.squeeze(0).squeeze(0)

        return emb_all