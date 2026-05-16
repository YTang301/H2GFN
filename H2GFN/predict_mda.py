import numpy as np
import copy
import pandas as pd
import os
import torch
from Input_MDA import All_Input_microRNA_disease
from training import H2GFN_all
from utils_model import gaussiansimilarity, integratedsimilarity

MDA,interaction,disease,miRNA,FS,FSP,SS,SSP = All_Input_microRNA_disease()
def train_all(interaction_original, FS, FSP, SS, SSP, args):
    nd, nm = interaction_original.shape
    interaction = copy.deepcopy(interaction_original)
    print("start training")
    kd, km = gaussiansimilarity(interaction, nd, nm)
    sd, sm = integratedsimilarity(FS, FSP, SS, SSP, kd, km)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    interaction_tensor = torch.FloatTensor(interaction).to(device)
    sd_tensor = torch.FloatTensor(sd).to(device)
    sm_tensor = torch.FloatTensor(sm).to(device)
    F = H2GFN_all(interaction_tensor.cpu().numpy(), sd_tensor.cpu().numpy(), sm_tensor.cpu().numpy(), args, device=device)
    print("done")
    return F


def case_study1(root0, args, output_path="Final_Prediction_Results.xlsx"):
    F = train_all(interaction, FS, FSP, SS, SSP, args)
    df_d = pd.read_excel(os.path.join(root0, 'Disease_Name.xlsx'),
                         header=None, names=['Num', 'Name'])
    df_m = pd.read_excel(os.path.join(root0, 'miRNA_Name.xlsx'),
                         header=None, names=['Num', 'Name'])
    disease_names = df_d['Name'].values
    mirna_names = df_m['Name'].values
    nd, nm = F.shape

    if len(disease_names) != nd or len(mirna_names) != nm:
        disease_names = disease_names[:nd]
        mirna_names = mirna_names[:nm]
        print("warning")

    df_result = pd.DataFrame({
        "Disease Name": np.repeat(disease_names, len(mirna_names)),
        "miRNA Name": np.tile(mirna_names, len(disease_names)),
        "Score": F.flatten()
    })
    df_result = df_result.sort_values(by="Score", ascending=False)
    df_result.to_excel(output_path, index=False)
    print(f"Excel：{output_path}")