from GMSTL import GMSTL
from layers import *
from BiRGC import BiRGC


class H2GFN(nn.Module):
    def __init__(
            self,
            gcn_layer1_units,
            gcn_layer2_units,
            gcn_threshold,
            rwr_c,
            gcn1_innum,
            hops,
            output_dim,
            input_dim,
            pe_dim,
            num_dis,
            num_mirna,
            graphformer_layers,
            num_heads,
            hidden_dim,
            ffn_dim,
            dropout_rate,
            GCNII_layers,
            alpha,
            theta
    ):
        super().__init__()
        self.seq_len = hops + 1
        self.pe_dim = int(pe_dim)
        self.input_dim = input_dim
        self.hidden_dim = int(hidden_dim)
        self.output_dim = output_dim
        self.ffn_dim = int(ffn_dim)
        self.num_heads = int(num_heads)
        self.graphformer_layers = graphformer_layers
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.num_dis = num_dis
        self.num_mirna = num_mirna
        self.alpha = alpha
        self.theta = theta

        self.birgc = BiRGC(
            input_dim=self.input_dim,
            pe_dim=self.pe_dim,
            gcnii_layers=GCNII_layers,
            alpha=self.alpha,
            theta=self.theta
        )

        self.gmstl = GMSTL(
            muti_in_dim1=self.input_dim,
            muti_in_dim2=self.pe_dim,
            num_dis=num_dis,
            num_mirna=num_mirna,
            layer1_hidden_units=gcn_layer1_units,
            layer2_hidden_units=gcn_layer2_units,
            threshold=gcn_threshold,
            rwr_c=rwr_c,
            gcn1_innum=gcn1_innum,
        )

        total_dim = 2*input_dim + gcn_layer1_units+gcn_layer2_units
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.output_dim),
        )
        self.decoder = InnerProductDecoder(self.output_dim, self.dropout_rate, self.num_dis)
        self.apply(lambda module: init_params(module, n_layers=self.graphformer_layers))

    def forward(self, dis_data, mirna_data, adj, feature):
        x_BiRGC = self.birgc(dis_data, mirna_data)
        x_GMSTL = self.gmstl(feature, adj=adj)
        output = torch.cat((feature, x_BiRGC, x_GMSTL), dim=1)
        embeddings = self.mlp(output)
        x1 = self.decoder(embeddings)
        return x1