import numpy as np
import pandas as pd
import os

def All_Input_microRNA_disease():
    '''
    root0 = '/mnt/rgcgt1022/H2GFN/microRNA-disease'
    disease = pd.read_excel(os.path.join(root0, 'miRNA-disease association/disease name.xlsx'), header=None, names=['Num','Name'])
    miRNA = pd.read_excel(os.path.join(root0, 'miRNA-disease association/miRNA name.xlsx'), header=None, names=['Num','Name'])
    MDA = np.loadtxt(os.path.join(root0, 'miRNA-disease association/knowndiseasemirnainteraction.txt'), dtype=int)

    nd = max(MDA[:, 1])
    nm = max(MDA[:, 0])
    rows, columns = MDA.shape
    SS1 = np.loadtxt(os.path.join(root0, 'disease semantic similarity 1/disease semantic similarity matrix 1.txt'))
    SS = SS1

    FS = np.loadtxt(os.path.join(root0, 'miRNA functional simialrity/functional similarity matrix.txt'))
    FSP = np.loadtxt(os.path.join(root0, 'miRNA functional simialrity/Functional similarity weighting matrix.txt'),
                     dtype=int)
    SSP = np.loadtxt(os.path.join(root0, 'disease semantic similarity 1/disease semantic similarity weighting matrix1.txt'),
        dtype=int)
    interaction = np.zeros((nd, nm), dtype=int)

    for i in range(0, rows):
        interaction[MDA[i, 1]-1, MDA[i, 0]-1] = 1
    '''


    root0 = '/mnt/rgcgt1022/H2GFN/microRNA-disease/HMDD_V1'
    disease = pd.read_excel(os.path.join(root0, 'Disease_Name.xlsx'), header=None,
                            names=['Num', 'Name'])
    miRNA = pd.read_excel(os.path.join(root0, 'miRNA_Name.xlsx'), header=None,
                          names=['Num', 'Name'])
    MDA = np.loadtxt(os.path.join(root0, 'HMDD1.txt'), dtype=int)
    nd = max(MDA[:, 1]) 
    nm = max(MDA[:, 0]) 
    rows, columns = MDA.shape

    SS = np.loadtxt(os.path.join(root0,'DisSim.txt'))
    FS = np.loadtxt(os.path.join(root0, 'miRSim.txt'))
    FSP = np.loadtxt(os.path.join(root0, 'miRSim_weight.txt'),dtype=int)

    SSP = np.loadtxt(os.path.join(root0, 'DisSim_weight.txt'),dtype=int)
    interaction = np.zeros((nd, nm), dtype=int)
    for i in range(0, rows):
        interaction[MDA[i, 1]-1, MDA[i, 0]-1] = 1


    '''
    root0 = '/mnt/rgcgt1022/H2GFN/microRNA-disease/HMDD_v3.2'
    disease = pd.read_excel(os.path.join(root0, 'disease_name_374.xlsx'), header=None,names=['Num', 'Name'])
    miRNA = pd.read_excel(os.path.join(root0, 'miRNA_name_788.xlsx'), header=None,names=['Num', 'Name'])
  
    interaction = np.loadtxt(os.path.join(root0, 'miRNA_disease_association.txt'), dtype=int)
    interaction = interaction.T

    nd, nm = interaction.shape
    MDA = np.zeros((sum(sum(interaction)), 2))
    n = 0

    for i in range(0, nd): 
        for j in range(0, nm):
            if interaction[i, j] == 1:
                MDA[n, 1] = i + 1 
                MDA[n, 0] = j + 1
                n = n + 1

    MDA = MDA.astype(np.int64)
    SS = np.loadtxt(os.path.join(root0, 'disease_semantic_sim.txt'))
    FS1 = np.loadtxt(os.path.join(root0, 'miRNA_semantic_sim.txt'))
    FS2 = np.loadtxt(os.path.join(root0, 'miRNA_sequence_sim.txt'))
    FS = (FS1 + FS2) / 2
    FSP = FS
    FSP[FSP > 0] = 1
    SSP = SS
    SSP[SSP > 0] = 1
    '''
    diseaseName = disease.Name
    miRNAName = miRNA.Name

    return MDA, interaction, diseaseName, miRNAName, FS,FSP, SS,SSP
