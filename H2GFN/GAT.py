import torch
import torch.nn as nn
import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.ops import edge_softmax

class GAT(nn.Module):
    def __init__(self, args, X_indim, X_outdim, num_hid, num_layers, num_outheads, encoding=False):
        super(GAT, self).__init__()
        self.args = args
        self.num_hid = num_hid
        self.num_heads = self.args.maed_heads
        self.num_outheads = num_outheads
        self.num_layers = num_layers
        self.in_drop = self.args.in_drop
        self.attn_drop = self.args.attn_drop
        self.negative_slope = self.args.negative_slope
        self.gat_layers = nn.ModuleList()
        self.activation = nn.PReLU()
        self.residual = self.args.residual
        self.norm = self.args.norm

        last_activation = self.activation if encoding else None
        last_residual = (encoding and self.residual)
        last_norm = self.norm if encoding else None

        if num_layers == 1:
            self.gat_layers.append(GATsingle(
                X_indim, X_outdim, self.num_outheads,
                self.in_drop, self.attn_drop, self.negative_slope, last_residual))
        else:
            self.gat_layers.append(GATsingle(
                X_indim, self.num_hid, self.num_heads,
                self.in_drop, self.attn_drop, self.negative_slope, self.residual, activation=nn.PReLU()))

            for l in range(1, num_layers - 1):
                self.gat_layers.append(GATsingle(
                    self.num_hid * self.num_heads, self.num_hid, self.num_heads,
                    self.in_drop, self.attn_drop, self.negative_slope, self.residual, activation=nn.PReLU()))

            self.gat_layers.append(GATsingle(
                self.num_hid * self.num_heads, X_outdim, self.num_outheads,
                self.in_drop, self.attn_drop, self.negative_slope, last_residual, activation=last_activation))

        self.head = nn.Identity()

    def forward(self, G, pre_X, return_hidden=False):
        h = pre_X
        hidden_list = []

        for l in range(self.num_layers):
            h = self.gat_layers[l](G, h)
            hidden_list.append(h)

        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

class GATsingle(nn.Module):
    def __init__(self,
                 Xsgl_indim,
                 Xsgl_outdim,
                 num_heads,
                 in_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero=False,
                 bias=True,
                 norm=None,
                 concat_out=True):
        super(GATsingle, self).__init__()

        self.num_heads = num_heads
        self.src_indim, self.dst_indim = expand_as_pair(Xsgl_indim)
        self.Xsgl_outdim = Xsgl_outdim
        self.allow_zero = allow_zero
        self.concat_out = concat_out

        if isinstance(Xsgl_indim, tuple):
            self.fc_src = nn.Linear(self.src_indim, Xsgl_outdim * num_heads, bias=False)
            self.fc_dst = nn.Linear(self.dst_indim, Xsgl_outdim * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self.src_indim, Xsgl_outdim * num_heads, bias=False)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, Xsgl_outdim)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, Xsgl_outdim)))
        self.in_drop = nn.Dropout(in_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * Xsgl_outdim,)))
        else:
            self.register_buffer('bias', None)

        if residual:
            if self.dst_indim != Xsgl_outdim * num_heads:
                self.res_fc = nn.Linear(self.dst_indim, num_heads * Xsgl_outdim, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        self.reset_parameters()
        self.activation = activation
        self.norm = norm
        if norm is not None:
            self.norm = norm(num_heads * Xsgl_outdim)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            # Check for nodes with no incoming edges to avoid invalid outputs
            if not self.allow_zero:
                if (graph.in_degrees() == 0).any():
                    raise RuntimeError('There are 0-in-degree nodes in the graph. '
                                       'Add self-loops using `g = dgl.add_self_loop(g)`.')

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.in_drop(feat[0])
                h_dst = self.in_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(*src_prefix_shape, self.num_heads, self.Xsgl_outdim)
                    feat_dst = self.fc(h_dst).view(*dst_prefix_shape, self.num_heads, self.Xsgl_outdim)
                else:
                    feat_src = self.fc_src(h_src).view(*src_prefix_shape, self.num_heads, self.Xsgl_outdim)
                    feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape, self.num_heads, self.Xsgl_outdim)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.in_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self.num_heads, self.Xsgl_outdim)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            if self.bias is not None:
                rst = rst + self.bias.view(*((1,) * len(dst_prefix_shape)), self.num_heads, self.Xsgl_outdim)

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self.Xsgl_outdim)
                rst = rst + resval
            if self.concat_out:
                rst = rst.flatten(1)
            else:
                rst = torch.mean(rst, dim=1)

            if self.norm is not None:
                rst = self.norm(rst)

            if self.activation:
                rst = self.activation(rst)

            return (rst, graph.edata['a']) if get_attention else rst