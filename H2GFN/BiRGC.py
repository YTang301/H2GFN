import torch
import torch.nn as nn
from layers import GCNIIConv

class BiRGC(nn.Module):
    def __init__(self, input_dim, pe_dim, gcnii_layers,alpha,theta):
        super(BiRGC, self).__init__()
        self.gcnii_layers = int(gcnii_layers)
        self.conv_channels = int(input_dim)
        self.convs = nn.ModuleList()
        self.alpha = alpha
        self.theta = theta
        for i in range(self.gcnii_layers):
            conv = GCNIIConv(channels=self.conv_channels, alpha=self.alpha, theta=self.theta, layer=i + 1)
            self.convs.append(conv)

    def forward(self, dis_data, mirna_data):
        x0_dis = dis_data.x
        x_dis = x0_dis
        for conv in self.convs:
            x_dis = conv(x_dis, x0_dis, dis_data.edge_index)

        x0_mirna = mirna_data.x
        x_mirna = x0_mirna
        for conv in self.convs:
            x_mirna = conv(x_mirna, x0_mirna, mirna_data.edge_index)

        x_Birgc = torch.cat((x_dis, x_mirna), dim=0)
        return x_Birgc