from layers import *
import torch.nn.functional as F
import random
from model_main import H2GFN
from MAED import MAED
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from utils_model import constructNet, PolynomialDecayLR, get_link_labels
import torch

def H2GFN_all(association_matrix, dis_simi_matrix, mirna_simi_matrix, args, device=None, ):
    epochs = int(args.all_epochs)
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_dis, num_mirna = association_matrix.shape
    Adj = pd.DataFrame(association_matrix)
    dis_simi = dis_simi_matrix
    dis_adj = np.where(dis_simi > args.gamma, 1, 0)
    dis_adj = torch.tensor(dis_adj, dtype=torch.int8).to(device)
    mirna_simi = mirna_simi_matrix
    mirna_adj = np.where(mirna_simi > args.gamma, 1, 0)
    mirna_adj = torch.tensor(mirna_adj, dtype=torch.int8).to(device)

    nd, nm = association_matrix.shape
    MDA = np.zeros((int(sum(sum(association_matrix))), 2))
    n = 0

    for i in range(0, nd):
        for j in range(0, nm):
            if association_matrix[i, j] == 1:
                MDA[n, 1] = i + 1
                MDA[n, 0] = j + 1
                n = n + 1

    node_input_dim = int(args.main_feature_dim)
    new_X = MAED(MDA, association_matrix, dis_simi_matrix, mirna_simi_matrix, args)

    name1 = 'disease'
    name2 = 'miRNA'
    dis_maed = new_X[name1].detach().cpu().numpy()
    mirna_maed = new_X[name2].detach().cpu().numpy()

    pca = PCA(n_components=node_input_dim)
    PCA_dis_feature = pca.fit_transform(dis_maed)
    PCA_mirna_feature = pca.fit_transform(mirna_maed)

    dis_feature = torch.FloatTensor(PCA_dis_feature).to(device)
    mirna_feature = torch.FloatTensor(PCA_mirna_feature).to(device)
    feature = torch.cat((dis_feature, mirna_feature), dim=0).to(device)
    Or_train_matrix = association_matrix.copy()
    or_adj = constructNet(torch.tensor(Or_train_matrix)).to(device)
    dis_network = torch.nonzero(dis_adj, as_tuple=True)
    dis_network = torch.stack(dis_network)
    dis_data = Data(x=feature[:num_dis,], edge_index=dis_network).to(device)
    mirna_network = torch.nonzero(mirna_adj, as_tuple=True)
    mirna_network = torch.stack(mirna_network)
    mirna_data = Data(x=feature[num_dis:,], edge_index=mirna_network).to(device)

    model = H2GFN(
        gcn_layer1_units=args.gcn_layer1_units,
        gcn_layer2_units=args.gcn_layer2_units,
        gcn_threshold=args.gcn_threshold,
        rwr_c=args.rwr_c,
        gcn1_innum=args.gcn1_innum,
        hops=int(args.hops),
        output_dim=64,
        input_dim=int(args.main_feature_dim),
        pe_dim=int(args.pe_dim),
        num_dis=num_dis,
        num_mirna=num_mirna,
        graphformer_layers=1,
        num_heads=int(args.num_heads),
        hidden_dim=int(args.hidden_dim),
        ffn_dim=int(args.ffn_dim),
        dropout_rate=args.dropout,
        GCNII_layers=int(args.gcn2_layers),
        alpha=args.alpha,
        theta=args.theta
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-4
    )
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=400,
        tot_updates=1000,
        lr=0.001,
        end_lr=0.0001,
        power=1.0
    )
    criterion = F.binary_cross_entropy
    epoch_aucs = []

    model.train()
    for epoch in range(epochs):
        train_pos_edge_index = np.asmatrix(np.where(Or_train_matrix > 0))
        train_pos_edge_index = torch.tensor(train_pos_edge_index, dtype=torch.long).to(device)
        train_neg_edge_index = np.asmatrix(np.where(Or_train_matrix < 1)).T.tolist()
        random.shuffle(train_neg_edge_index)
        train_neg_edge_index = train_neg_edge_index[:train_pos_edge_index.shape[1]]
        train_neg_edge_index = np.array(train_neg_edge_index).T
        train_neg_edge_index = torch.tensor(train_neg_edge_index, dtype=torch.long).to(device)

        output = model(
            dis_data=dis_data.to(device),
            mirna_data=mirna_data.to(device),
            adj=or_adj.to(device),
            feature=feature.to(device)
        )

        edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], 1).to(device)
        scores = output[edge_index[0], edge_index[1]]
        labels = get_link_labels(train_pos_edge_index, train_neg_edge_index).to(device)
        loss = criterion(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        y_true = labels.cpu().detach().numpy()
        y_score = scores.cpu().detach().numpy()
        current_auc = roc_auc_score(y_true, y_score)
        epoch_aucs.append(current_auc)

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred_matrix = model(
            dis_data=dis_data.to(device),
            mirna_data=mirna_data.to(device),
            adj=or_adj.to(device),
            feature=feature.to(device)
        )
    return pred_matrix.cpu().detach().numpy()