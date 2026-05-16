##  Project Overview

H2GFN is a novel homogeneous and heterogeneous graph fusion network designed to accurately predict miRNA-disease associations by integrating diverse biological information from various graph structures. This model employs a multi-perspective masked autoencoder and bilateral residual graph convolutions to extract robust representations, which are then fused with heterogeneous miRNA-disease association networks using a gating multi-scale topology learning module. Our approach effectively captures complex relationships and reduces noisy data, significantly enhancing prediction accuracy compared to state-of-the-art methods.

## Key Features

*   **Graph Fusion: Seamlessly integrates information from both homogeneous and heterogeneous graphs.
*   **Multi-perspective Representation Learning: Utilizes a masked autoencoder for robust feature extraction from multiple data perspectives.
*   **Bilateral Residual Graph Convolutions: Enhances information flow and addresses potential over-smoothing issues in deep graph networks.
*   **Gating Multi-scale Topology Learning: Dynamically fuses learned representations with the heterogeneous miRNA-disease association network.

## Our system meets the following requirements:
*   Python Version==3.12.4
*   Pytorch==2.3.1
*   CUDA==12.1
*   dgl==2.1.0+cu121
*   torch-geometric==2.6.1
*   torch_scatter==2.1.2+pt23cu121
*   torch_sparse==0.6.18+pt23cu121
*   scikit-learn==1.5.0
*   numpy==2.0.0
*   mpmath==1.3.0
*   pandas==2.2.2
*   networkx==3.2.1

## Data Preparation & Path Configuration：
Crucially, ensure your datasets are correctly placed and adjust the root path in the code as needed.

## Running the Model：
The H2GFN model is executed via the main.py script. You can run it for training and evaluation.
